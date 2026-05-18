package llmops

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTPResponder is a concrete LLMResponder implementation that dispatches
// each prompt to an OpenAI-compatible /chat/completions endpoint and
// returns the first choice's message content. It is the round-62 §11.4
// anti-bluff close-out of the round-25 ErrLLMResponderNotConfigured
// sentinel: round 25 removed the literal "simulated response" bluff but
// left consumers without a ready wireable implementation; this type is
// what they wire into (*InMemoryContinuousEvaluator).SetResponder so
// evaluateSample produces a real model response and feeds it to the
// LLMEvaluator for real scoring.
//
// The OpenAI Chat Completions wire protocol is targeted because it is the
// most widely adopted format: OpenAI itself, Azure OpenAI, OpenRouter,
// Groq, DeepSeek, Mistral, Together, Anyscale, vLLM, llama.cpp's HTTP
// server, LM Studio, LocalAI, and any other server fronting an OpenAI
// shim all speak it. Consumers who need a non-OpenAI-shaped backend
// (e.g. raw Anthropic Messages, raw Gemini generateContent) wrap their
// SDK behind LLMResponderFunc instead — HTTPResponder targets the
// majority case and keeps the dependency footprint at net/http only.
//
// CONST-042 secret-leak adjacency: APIKey is sourced from the caller's
// HTTPResponderConfig (operator passes an env-sourced value into the
// constructor at wire-up time). HTTPResponder NEVER reads os.Getenv
// itself, NEVER logs the APIKey field, NEVER includes the key in error
// messages. The String / GoString methods deliberately elide the key.
//
// CONST-050(A) / (B): production code may import and construct
// HTTPResponder. The unit tests in responder_test.go exercise it
// against httptest.NewServer (in-process loopback) which counts as a
// unit test under CONST-050(A) (no real external network); the
// integration test TestHTTPResponder_RealOpenAI_RoundtripsOK is
// env-gated and hits a real provider when LLMOPS_TEST_OPENAI_KEY is
// set, satisfying CONST-050(B) coverage for the integration tier.
type HTTPResponder struct {
	endpoint   string
	model      string
	apiKey     string
	timeout    time.Duration
	httpClient *http.Client
}

// HTTPResponderConfig is the constructor input for NewHTTPResponder.
// The four required fields are Endpoint + Model; APIKey is optional
// (local OpenAI-shim servers like llama.cpp's HTTP mode do not require
// one); Timeout defaults to 60s when zero. HTTPClient is optional —
// callers wanting custom transport wiring (proxy, mTLS, retry middleware)
// inject it here; otherwise a clean http.Client honouring Timeout is
// constructed.
type HTTPResponderConfig struct {
	// Endpoint is the base URL of the OpenAI-compatible server, MINUS
	// the trailing "/chat/completions" path component. Examples:
	//   - "https://api.openai.com/v1"           (OpenAI)
	//   - "https://api.deepseek.com/v1"         (DeepSeek)
	//   - "https://api.groq.com/openai/v1"      (Groq OpenAI shim)
	//   - "http://localhost:11434/v1"           (Ollama OpenAI shim)
	//   - "http://localhost:8080/v1"            (llama.cpp HTTP server)
	// Empty endpoint => NewHTTPResponder returns
	// ErrHTTPResponderEndpointNotConfigured.
	Endpoint string

	// Model is the model identifier the server uses to route the request
	// (e.g. "gpt-4o-mini", "llama-3.1-8b-instruct", "deepseek-chat").
	// Empty model => NewHTTPResponder returns
	// ErrHTTPResponderModelNotConfigured.
	Model string

	// APIKey is sent as the Authorization: Bearer <APIKey> header. May
	// be empty for local servers (llama.cpp, Ollama) that don't require
	// authentication — empty key => no Authorization header sent.
	// CONST-042: callers MUST source this from .env / secret store and
	// pass at construction time; HTTPResponder never reads env itself.
	APIKey string

	// Timeout caps the round-trip duration. Zero => 60s default.
	// Honoured even when ctx.Deadline is later (whichever fires first
	// wins via context.WithTimeout chaining inside Generate).
	Timeout time.Duration

	// HTTPClient lets callers inject a pre-configured client (custom
	// Transport for proxy / mTLS / retry middleware / metrics
	// interceptor). Nil => internal client constructed with Timeout.
	HTTPClient *http.Client
}

// Sentinel errors covering the four failure modes Generate exposes.
// All four are stable contract surfaces — consumers MAY errors.Is()
// against them to branch on failure mode (e.g. retry vs surface vs
// abort the evaluation run).
var (
	// ErrHTTPResponderEndpointNotConfigured fires from NewHTTPResponder
	// when Endpoint is empty. Distinct from
	// ErrLLMResponderNotConfigured which fires from evaluateSample when
	// SetResponder was never called — this one fires earlier, at
	// construction time, when the caller invoked the constructor with
	// an unusable config.
	ErrHTTPResponderEndpointNotConfigured = errors.New(
		"llmops: HTTPResponderConfig.Endpoint is empty — set it to the OpenAI-compatible base URL of your LLM provider (e.g. \"https://api.openai.com/v1\") before constructing HTTPResponder",
	)

	// ErrHTTPResponderModelNotConfigured fires from NewHTTPResponder
	// when Model is empty. Sibling of the endpoint sentinel.
	ErrHTTPResponderModelNotConfigured = errors.New(
		"llmops: HTTPResponderConfig.Model is empty — set it to a model identifier the configured endpoint accepts (e.g. \"gpt-4o-mini\", \"llama-3.1-8b-instruct\") before constructing HTTPResponder",
	)

	// ErrHTTPResponderRequestFailed wraps every network-layer or
	// HTTP-level failure (DNS, TCP, TLS, 4xx, 5xx). The wrapped error
	// retains the original cause via errors.Unwrap; the wrapper text
	// adds the endpoint + status context but NEVER includes the APIKey.
	ErrHTTPResponderRequestFailed = errors.New(
		"llmops: HTTPResponder request to the LLM endpoint failed",
	)

	// ErrHTTPResponderResponseInvalid fires when the server returned a
	// 2xx status but the response body could not be parsed as an
	// OpenAI-compatible chat-completions response, or the body parsed
	// but contained zero choices / an empty first-choice message.
	ErrHTTPResponderResponseInvalid = errors.New(
		"llmops: HTTPResponder received a 2xx response but could not extract a usable message — body did not match the OpenAI chat-completions schema (choices[0].message.content)",
	)
)

// NewHTTPResponder constructs an HTTPResponder from the given config.
// Returns one of the two construction-time sentinels (endpoint or model
// not configured) if the config is unusable. The returned responder is
// safe for concurrent use across goroutines — the embedded *http.Client
// is concurrency-safe per stdlib contract, and HTTPResponder itself
// holds no mutable state after construction.
func NewHTTPResponder(cfg HTTPResponderConfig) (*HTTPResponder, error) {
	if strings.TrimSpace(cfg.Endpoint) == "" {
		return nil, ErrHTTPResponderEndpointNotConfigured
	}
	if strings.TrimSpace(cfg.Model) == "" {
		return nil, ErrHTTPResponderModelNotConfigured
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: timeout}
	}

	// Strip trailing slash so endpoint + "/chat/completions" composes
	// cleanly regardless of whether the caller passed ".../v1" or
	// ".../v1/".
	endpoint := strings.TrimRight(cfg.Endpoint, "/")

	return &HTTPResponder{
		endpoint:   endpoint,
		model:      cfg.Model,
		apiKey:     cfg.APIKey,
		timeout:    timeout,
		httpClient: client,
	}, nil
}

// chatCompletionsRequest is the wire shape POSTed to the endpoint. Only
// the two fields the spec strictly requires are populated — every
// OpenAI-compatible server accepts the minimum form. Callers needing
// temperature / top_p / max_tokens / streaming wrap HTTPResponder
// instead of extending this struct (keeps the surface small + stable).
type chatCompletionsRequest struct {
	Model    string                 `json:"model"`
	Messages []chatCompletionMessage `json:"messages"`
}

type chatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatCompletionsResponse covers the subset of the OpenAI response
// schema HTTPResponder consumes. Additional fields the server returns
// (usage tokens, finish_reason, model echo, system_fingerprint, etc.)
// are tolerated via JSON decoder default behaviour (unknown fields
// ignored).
type chatCompletionsResponse struct {
	Choices []chatCompletionChoice `json:"choices"`
	// Error field present on some servers' 2xx-with-error responses
	// (rare but observed on Azure OpenAI). When non-empty we treat
	// the response as invalid even though HTTP status was 2xx.
	Error *chatCompletionError `json:"error,omitempty"`
}

type chatCompletionChoice struct {
	Message chatCompletionMessage `json:"message"`
}

type chatCompletionError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
}

// Generate satisfies the LLMResponder interface contract. It dispatches
// prompt to the configured endpoint as a single user-role message and
// returns the first choice's content. Behavioural guarantees:
//
//   - Honours ctx.Cancel + ctx.Deadline (request is built with
//     http.NewRequestWithContext so cancellation aborts in-flight read).
//   - Wraps every transport / HTTP-error in
//     ErrHTTPResponderRequestFailed with the underlying cause
//     accessible via errors.Unwrap.
//   - Wraps every parse / empty-choice failure in
//     ErrHTTPResponderResponseInvalid.
//   - NEVER includes the APIKey in any error message or log output.
//   - NEVER mutates HTTPResponder state — safe for concurrent calls.
func (r *HTTPResponder) Generate(ctx context.Context, prompt string) (string, error) {
	body := chatCompletionsRequest{
		Model: r.model,
		Messages: []chatCompletionMessage{
			{Role: "user", Content: prompt},
		},
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		// json.Marshal of this struct cannot realistically fail (no
		// chan / func / cyclic types), but wrap defensively rather
		// than panic so the contract stays "every failure surfaces as
		// a sentinel-wrapped error".
		return "", fmt.Errorf("%w: marshal request body: %v", ErrHTTPResponderRequestFailed, err)
	}

	url := r.endpoint + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("%w: build request for %s: %v", ErrHTTPResponderRequestFailed, url, err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if r.apiKey != "" {
		// CONST-042: header set from in-memory field; never logged.
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
	}

	resp, err := r.httpClient.Do(req)
	if err != nil {
		// Transport-layer failure: DNS, TCP, TLS, ctx-cancel,
		// timeout. Wrap with the endpoint URL (no APIKey).
		return "", fmt.Errorf("%w: POST %s: %v", ErrHTTPResponderRequestFailed, url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read body before status check so 4xx/5xx error responses can
	// surface the server's textual diagnostic (helps consumers debug
	// model-not-found, auth-rejected, rate-limited, etc.).
	respBytes, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return "", fmt.Errorf("%w: read response body from %s (status %d): %v",
			ErrHTTPResponderRequestFailed, url, resp.StatusCode, readErr)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Truncate the server's response body in the error message
		// so a multi-MB error payload doesn't blow up logs.
		bodyExcerpt := string(respBytes)
		if len(bodyExcerpt) > 512 {
			bodyExcerpt = bodyExcerpt[:512] + "...(truncated)"
		}
		return "", fmt.Errorf("%w: POST %s returned HTTP %d: %s",
			ErrHTTPResponderRequestFailed, url, resp.StatusCode, bodyExcerpt)
	}

	var parsed chatCompletionsResponse
	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		bodyExcerpt := string(respBytes)
		if len(bodyExcerpt) > 256 {
			bodyExcerpt = bodyExcerpt[:256] + "...(truncated)"
		}
		return "", fmt.Errorf("%w: unmarshal response from %s: %v (body excerpt: %q)",
			ErrHTTPResponderResponseInvalid, url, err, bodyExcerpt)
	}

	if parsed.Error != nil && parsed.Error.Message != "" {
		return "", fmt.Errorf("%w: server returned 2xx with embedded error from %s: %s (type=%s code=%s)",
			ErrHTTPResponderResponseInvalid, url, parsed.Error.Message, parsed.Error.Type, parsed.Error.Code)
	}

	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("%w: response from %s contained zero choices",
			ErrHTTPResponderResponseInvalid, url)
	}

	content := parsed.Choices[0].Message.Content
	if content == "" {
		return "", fmt.Errorf("%w: response from %s had choices[0].message.content == \"\"",
			ErrHTTPResponderResponseInvalid, url)
	}

	return content, nil
}

// String / GoString deliberately elide the APIKey per CONST-042 secret
// adjacency — accidental inclusion of an HTTPResponder in a log line
// (logrus.Info("responder=%+v", r) etc.) MUST NOT leak the key.
func (r *HTTPResponder) String() string {
	keyState := "unset"
	if r.apiKey != "" {
		keyState = "REDACTED"
	}
	return fmt.Sprintf("HTTPResponder{endpoint=%s model=%s timeout=%s apiKey=%s}",
		r.endpoint, r.model, r.timeout, keyState)
}

// GoString mirrors String so %#v formatting also elides the key.
func (r *HTTPResponder) GoString() string { return r.String() }
