package llmops

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Round-62 §11.4 anti-bluff close-out of the round-25
// ErrLLMResponderNotConfigured sentinel. The tests below exercise the
// concrete HTTPResponder implementation introduced in responder.go.
//
// Anti-bluff posture: every test that claims HTTPResponder "works"
// either (a) drives a real httptest.Server and asserts on the exact
// body it returned, or (b) is env-gated against a real OpenAI endpoint
// and skips with SKIP-OK when the key is absent. No test asserts only
// "no error returned" — every PASS path asserts on a concrete value
// the responder produced from a concrete server response.

// --- Construction-time sentinel tests -----------------------------------

func TestNewHTTPResponder_EmptyEndpoint_ReturnsSentinel(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "",
		Model:    "gpt-4o-mini",
	})
	require.Error(t, err, "empty endpoint MUST return sentinel, not a usable responder")
	assert.Nil(t, r, "responder MUST be nil when construction fails")
	assert.ErrorIs(t, err, ErrHTTPResponderEndpointNotConfigured,
		"err MUST wrap ErrHTTPResponderEndpointNotConfigured so callers can errors.Is-branch")
}

func TestNewHTTPResponder_WhitespaceEndpoint_ReturnsSentinel(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "   \t  \n  ",
		Model:    "gpt-4o-mini",
	})
	require.Error(t, err, "whitespace-only endpoint MUST be treated as empty")
	assert.Nil(t, r)
	assert.ErrorIs(t, err, ErrHTTPResponderEndpointNotConfigured)
}

func TestNewHTTPResponder_EmptyModel_ReturnsSentinel(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "https://api.openai.com/v1",
		Model:    "",
	})
	require.Error(t, err, "empty model MUST return sentinel")
	assert.Nil(t, r)
	assert.ErrorIs(t, err, ErrHTTPResponderModelNotConfigured)
}

func TestNewHTTPResponder_ValidConfig_StripsTrailingSlash(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "https://api.openai.com/v1/",
		Model:    "gpt-4o-mini",
	})
	require.NoError(t, err)
	require.NotNil(t, r)
	assert.Equal(t, "https://api.openai.com/v1", r.endpoint,
		"trailing slash MUST be stripped so endpoint + \"/chat/completions\" composes cleanly")
}

func TestNewHTTPResponder_ZeroTimeout_AppliesDefault(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "https://api.openai.com/v1",
		Model:    "gpt-4o-mini",
	})
	require.NoError(t, err)
	assert.Equal(t, 60*time.Second, r.timeout, "zero Timeout MUST default to 60s")
}

func TestNewHTTPResponder_String_RedactsAPIKey(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "https://api.openai.com/v1",
		Model:    "gpt-4o-mini",
		APIKey:   "sk-secret-must-never-appear-in-string-output",
	})
	require.NoError(t, err)

	s := r.String()
	assert.NotContains(t, s, "sk-secret-must-never-appear-in-string-output",
		"CONST-042: String() MUST NOT leak the APIKey")
	assert.Contains(t, s, "REDACTED",
		"String() MUST advertise that a key is set without disclosing it")

	gs := r.GoString()
	assert.NotContains(t, gs, "sk-secret-must-never-appear-in-string-output",
		"CONST-042: GoString() MUST NOT leak the APIKey (covers %#v formatting)")
}

func TestNewHTTPResponder_String_UnsetKeyShownAsUnset(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "http://localhost:11434/v1",
		Model:    "llama-3.1-8b",
	})
	require.NoError(t, err)
	assert.Contains(t, r.String(), "apiKey=unset",
		"missing key MUST render as \"unset\" so operators can distinguish absent vs redacted")
}

// --- Generate happy-path tests ------------------------------------------

// TestHTTPResponder_Generate_RoundtripsViaHTTPTest drives a real
// in-process httptest.Server, asserts the request shape POSTed by
// HTTPResponder matches the OpenAI chat-completions schema, and asserts
// the response content extracted by Generate equals the canned value
// the server returned. This is the core anti-bluff test for the
// "HTTPResponder actually works" claim — if this test passes, the
// responder genuinely roundtrips a prompt through an HTTP endpoint and
// extracts the content correctly. CONST-035 / Article XI §11.9.
func TestHTTPResponder_Generate_RoundtripsViaHTTPTest(t *testing.T) {
	var (
		gotPath   string
		gotMethod string
		gotAuth   string
		gotCT     string
		gotBody   chatCompletionsRequest
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		gotPath = req.URL.Path
		gotMethod = req.Method
		gotAuth = req.Header.Get("Authorization")
		gotCT = req.Header.Get("Content-Type")

		bodyBytes, err := io.ReadAll(req.Body)
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(bodyBytes, &gotBody))

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"choices": [
				{"message": {"role": "assistant", "content": "Paris is the capital of France."}}
			]
		}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL + "/v1",
		Model:    "test-model-round62",
		APIKey:   "test-key-please-show-up-in-bearer-header",
		Timeout:  5 * time.Second,
	})
	require.NoError(t, err)

	content, err := r.Generate(context.Background(), "What is the capital of France?")
	require.NoError(t, err, "happy-path Generate MUST succeed")

	// Content assertion: PROVES the responder really extracted what
	// the server returned (not a hardcoded literal in responder.go).
	assert.Equal(t, "Paris is the capital of France.", content,
		"Generate MUST return choices[0].message.content verbatim")

	// Request-shape assertions: PROVE HTTPResponder POSTs the right
	// thing per OpenAI chat-completions schema.
	assert.Equal(t, http.MethodPost, gotMethod, "MUST POST")
	assert.Equal(t, "/v1/chat/completions", gotPath, "MUST append /chat/completions to base URL")
	assert.Equal(t, "Bearer test-key-please-show-up-in-bearer-header", gotAuth,
		"MUST send APIKey as Bearer in Authorization header")
	assert.Contains(t, gotCT, "application/json", "MUST set Content-Type to application/json")
	assert.Equal(t, "test-model-round62", gotBody.Model, "MUST send configured model in body")
	require.Len(t, gotBody.Messages, 1, "MUST send exactly one user message")
	assert.Equal(t, "user", gotBody.Messages[0].Role, "message role MUST be \"user\"")
	assert.Equal(t, "What is the capital of France?", gotBody.Messages[0].Content,
		"message content MUST equal the prompt argument verbatim")
}

func TestHTTPResponder_Generate_NoAPIKey_OmitsAuthHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		gotAuth = req.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "local-model",
		// APIKey deliberately empty — covers local-server case (llama.cpp, Ollama).
	})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "ping")
	require.NoError(t, err)
	assert.Empty(t, gotAuth,
		"empty APIKey MUST result in no Authorization header (local servers reject unexpected auth)")
}

// --- Generate failure-mode tests ----------------------------------------

func TestHTTPResponder_Generate_500_ReturnsRequestFailedSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"backend overloaded"}}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
	})
	require.NoError(t, err)

	content, err := r.Generate(context.Background(), "anything")
	require.Error(t, err, "5xx MUST surface as error")
	assert.Empty(t, content)
	assert.ErrorIs(t, err, ErrHTTPResponderRequestFailed,
		"5xx MUST wrap ErrHTTPResponderRequestFailed")
	assert.Contains(t, err.Error(), "500", "error message MUST contain HTTP status code")
	assert.Contains(t, err.Error(), "backend overloaded",
		"error MUST include server body excerpt so debugging is possible")
}

func TestHTTPResponder_Generate_401_ReturnsRequestFailedSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid API key"}}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
		APIKey:   "bad-key",
	})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "anything")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrHTTPResponderRequestFailed)
	assert.Contains(t, err.Error(), "401")
	assert.NotContains(t, err.Error(), "bad-key",
		"CONST-042: 401 error message MUST NOT include the rejected APIKey")
}

func TestHTTPResponder_Generate_InvalidJSON_ReturnsResponseInvalidSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{this is not valid json`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
	})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "anything")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrHTTPResponderResponseInvalid,
		"unparseable body MUST wrap ErrHTTPResponderResponseInvalid")
}

func TestHTTPResponder_Generate_EmptyChoices_ReturnsResponseInvalidSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
	})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "anything")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrHTTPResponderResponseInvalid)
	assert.Contains(t, err.Error(), "zero choices")
}

func TestHTTPResponder_Generate_EmptyMessageContent_ReturnsResponseInvalidSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":""}}]}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
	})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "anything")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrHTTPResponderResponseInvalid)
	assert.Contains(t, err.Error(), "content == \"\"")
}

func TestHTTPResponder_Generate_2xxWithEmbeddedError_ReturnsResponseInvalidSentinel(t *testing.T) {
	// Some servers (notably Azure OpenAI in a content-filter scenario)
	// return HTTP 200 with an `error` object in the body. Treat as invalid.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"error":{"message":"content filter triggered","type":"policy","code":"filter_block"}}`))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
	})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "anything")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrHTTPResponderResponseInvalid)
	assert.Contains(t, err.Error(), "content filter triggered")
	assert.Contains(t, err.Error(), "policy")
	assert.Contains(t, err.Error(), "filter_block")
}

func TestHTTPResponder_Generate_HonoursContextCancel(t *testing.T) {
	var serverGotCancelled atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		select {
		case <-req.Context().Done():
			serverGotCancelled.Store(true)
		case <-time.After(2 * time.Second):
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"too late"}}]}`))
		}
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "m",
		Timeout:  5 * time.Second, // longer than the cancel deadline below
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err = r.Generate(ctx, "anything")
	elapsed := time.Since(start)

	require.Error(t, err, "ctx-cancel MUST surface as error")
	assert.ErrorIs(t, err, ErrHTTPResponderRequestFailed,
		"ctx-cancel MUST wrap ErrHTTPResponderRequestFailed (transport-layer error)")
	assert.Less(t, elapsed, 1500*time.Millisecond,
		"ctx-cancel MUST abort within ~deadline, not wait for the 2s server timer")
}

func TestHTTPResponder_Generate_LargeErrorBody_GetsTruncated(t *testing.T) {
	// Defensive truncation prevents a hostile server from spamming logs
	// with a multi-MB error payload.
	hugeBody := strings.Repeat("X", 10000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(hugeBody))
	}))
	defer srv.Close()

	r, err := NewHTTPResponder(HTTPResponderConfig{Endpoint: srv.URL, Model: "m"})
	require.NoError(t, err)

	_, err = r.Generate(context.Background(), "anything")
	require.Error(t, err)
	msg := err.Error()
	assert.Less(t, len(msg), 2000, "error message MUST stay bounded even when server body is huge")
	assert.Contains(t, msg, "...(truncated)", "huge body MUST be marked as truncated")
}

// --- Integration with evaluator close-out -------------------------------

// TestEvaluator_WithHTTPResponder_EndToEnd is the closing assertion for
// round-62 §11.4: it constructs the full StartRun pipeline with a real
// HTTPResponder pointing at an httptest.Server, runs an evaluation,
// and asserts that the SampleResult.Actual field equals the response
// the server returned. This proves the round-25 SetResponder seam ends
// in a real LLM dispatch when HTTPResponder is wired in — completing
// the close-out of ErrLLMResponderNotConfigured at the consumer level.
func TestEvaluator_WithHTTPResponder_EndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"42 is the answer"}}]}`))
	}))
	defer srv.Close()

	responder, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "test-evaluator-round62",
		Timeout:  5 * time.Second,
	})
	require.NoError(t, err)

	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel) // quiet test output
	eval := NewInMemoryContinuousEvaluator(
		&mockLLMEvaluator{scores: map[string]float64{"correctness": 0.95}},
		nil, nil, logger,
	)
	eval.SetResponder(responder) // round-62 close-out: real HTTPResponder, not echoResponder

	ctx := context.Background()
	ds := &Dataset{Name: "round62-end-to-end"}
	require.NoError(t, eval.CreateDataset(ctx, ds))
	require.NoError(t, eval.AddSamples(ctx, ds.ID, []*DatasetSample{
		{Input: "What is the meaning of life?", ExpectedOutput: "42"},
	}))

	run := &EvaluationRun{Name: "round62-run", Dataset: ds.ID, Metrics: []string{"correctness"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, run.ID))

	// Poll for completion (StartRun is async per CLAUDE.md gotcha #1).
	deadline := time.Now().Add(5 * time.Second)
	var got *EvaluationRun
	for time.Now().Before(deadline) {
		got, err = eval.GetRun(ctx, run.ID)
		require.NoError(t, err)
		if got.Status == EvaluationStatusCompleted {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	require.Equal(t, EvaluationStatusCompleted, got.Status,
		"run MUST complete within deadline when HTTPResponder is wired in")
	require.NotNil(t, got.Results)
	require.Len(t, got.Results.SampleResults, 1)

	// CORE ASSERTION of round-62: SampleResult.Actual is what the
	// server returned, NOT the literal "simulated response" the
	// round-25 bluff used to produce. Proves the whole pipeline now
	// goes through a real LLM dispatch path.
	assert.Equal(t, "42 is the answer", got.Results.SampleResults[0].Actual,
		"SampleResult.Actual MUST equal the LLM server's response — closes round-25 simulated-response bluff")
	assert.True(t, got.Results.SampleResults[0].Passed,
		"sample MUST PASS (0.95 score > 0.7 floor)")
	assert.Equal(t, 1.0, got.Results.PassRate)
}

// --- Real-provider integration test (env-gated) -------------------------

// TestHTTPResponder_RealOpenAI_RoundtripsOK is the CONST-050(B)
// integration tier coverage for HTTPResponder. When
// LLMOPS_TEST_OPENAI_KEY is set, it POSTs to api.openai.com with a
// trivial deterministic prompt and asserts a non-empty response —
// proving HTTPResponder really does roundtrip against a real OpenAI
// endpoint, not just an httptest.Server. The base URL + model are
// also overridable via env so the test can target any OpenAI-compatible
// provider (DeepSeek, Groq, OpenRouter, local llama.cpp, etc.).
//
// SKIP-OK: #LLMOPS-RESPONDER-REAL-ROUND62 — without the env key we
// cannot drive a real provider; the skip is loud per CONST-035 skip-bluff
// rule and the gate marker tracks the coverage gap.
func TestHTTPResponder_RealOpenAI_RoundtripsOK(t *testing.T) {
	apiKey := os.Getenv("LLMOPS_TEST_OPENAI_KEY")
	if apiKey == "" {
		t.Skip("SKIP-OK: #LLMOPS-RESPONDER-REAL-ROUND62 — LLMOPS_TEST_OPENAI_KEY not set; set to a real OpenAI-compatible key to exercise this integration tier")
	}

	endpoint := os.Getenv("LLMOPS_TEST_OPENAI_ENDPOINT")
	if endpoint == "" {
		endpoint = "https://api.openai.com/v1"
	}
	model := os.Getenv("LLMOPS_TEST_OPENAI_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}

	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: endpoint,
		Model:    model,
		APIKey:   apiKey,
		Timeout:  30 * time.Second,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	content, err := r.Generate(ctx, "Respond with exactly the word OK and nothing else.")
	require.NoError(t, err, "real-provider Generate MUST succeed; check LLMOPS_TEST_OPENAI_KEY validity")
	assert.NotEmpty(t, content, "real provider MUST return non-empty content")

	// Print to stdout so operator running with -v gets evidence the
	// real roundtrip happened (CONST-035 captured-evidence floor).
	t.Logf("real-provider roundtrip evidence (endpoint=%s model=%s): response=%q",
		endpoint, model, content)
	fmt.Fprintf(io.Discard, "consumed: %s\n", content) // also use the import even when -v absent
}
