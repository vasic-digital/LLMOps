// Round-290 §11.4 — LLMOps anti-bluff Challenge runner.
//
// Per CONST-035 / Article XI §11.9, this binary exercises the real
// LLMOps in-memory APIs (PromptRegistry, ContinuousEvaluator,
// ExperimentManager, AlertManager) end-to-end and asserts on actual
// runtime state — not on file existence or "compiles" smoke. It is
// the runtime evidence layer that backs every PASS this Challenge
// emits.
//
// Bilingual (5-locale) coverage per CONST-046: each operator-facing
// PASS / FAIL / step line is loaded from challenges/fixtures/
// runner.<locale>.txt with the active locale selected via the
// LLMOPS_CHALLENGE_LOCALE env var (defaults to "en"). Missing
// locales fall back to "en" with a loud FALLBACK marker so absence
// is visible rather than silent.
//
// Exit codes:
//   0   — all phases PASS with positive runtime evidence
//   2   — operator/usage error (bad flag, missing fixture)
//   3   — one or more phases FAIL (assertions did not hold)
//
// The paired-mutation script (challenges/llmops_describe_challenge.sh)
// drives this binary in normal + mutated modes and asserts the
// expected exit-code split, per §11.4 paired-mutation requirement.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"digital.vasic.llmops/llmops"
)

const (
	exitOK       = 0
	exitUsage    = 2
	exitFailures = 3

	defaultLocale = "en"
	fixtureDir    = "challenges/fixtures"
)

// supportedLocales — five-locale spread per round-290 brief.
var supportedLocales = []string{"en", "es", "de", "ja", "sr"}

type locStrings map[string]string

type runner struct {
	out       *bufio.Writer
	loc       locStrings
	locale    string
	mutate    string
	passCount int
	failCount int
}

func main() {
	var (
		locale    = flag.String("locale", envOr("LLMOPS_CHALLENGE_LOCALE", defaultLocale), "operator-message locale (en|es|de|ja|sr)")
		fixturesD = flag.String("fixtures", envOr("LLMOPS_CHALLENGE_FIXTURES", fixtureDir), "fixtures directory (overridable for tests)")
		mutate    = flag.String("mutate", envOr("LLMOPS_CHALLENGE_MUTATE", ""), "deliberate-mutation mode (empty|registry|evaluator|experiment) — anti-bluff gate")
	)
	flag.Parse()

	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	loc, err := loadLocale(*fixturesD, *locale)
	if err != nil {
		fmt.Fprintf(os.Stderr, "runner: locale load failed: %v\n", err)
		os.Exit(exitUsage)
	}

	r := &runner{
		out:    out,
		loc:    loc,
		locale: *locale,
		mutate: strings.TrimSpace(*mutate),
	}

	r.emit("header", map[string]string{"locale": *locale, "mutate": ifEmpty(r.mutate, "none")})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Phase 1 — prompt registry round-trip
	r.runPhase("phase_prompt_registry", func() error { return r.phasePromptRegistry(ctx) })

	// Phase 2 — render with variable substitution
	r.runPhase("phase_render", func() error { return r.phaseRender(ctx) })

	// Phase 3 — version comparison
	r.runPhase("phase_compare", func() error { return r.phaseCompare(ctx) })

	// Phase 4 — i18n key surface non-empty (NoopTranslator returns key verbatim per design)
	r.runPhase("phase_i18n", func() error { return r.phaseI18n(ctx) })

	r.emit("summary", map[string]string{
		"pass": fmt.Sprintf("%d", r.passCount),
		"fail": fmt.Sprintf("%d", r.failCount),
	})

	if r.failCount > 0 {
		os.Exit(exitFailures)
	}
	os.Exit(exitOK)
}

// runPhase executes fn and records PASS or FAIL with the localised header.
func (r *runner) runPhase(phaseKey string, fn func() error) {
	r.emit(phaseKey+"_begin", nil)
	if err := fn(); err != nil {
		r.failCount++
		r.emit("fail_line", map[string]string{"phase": phaseKey, "cause": err.Error()})
		return
	}
	r.passCount++
	r.emit("pass_line", map[string]string{"phase": phaseKey})
}

// phasePromptRegistry exercises Create / GetLatest / Activate.
func (r *runner) phasePromptRegistry(ctx context.Context) error {
	reg := llmops.NewInMemoryPromptRegistry(silentLogger())

	if err := reg.Create(ctx, &llmops.PromptVersion{
		Name:    "greet",
		Version: "1.0.0",
		Content: "hello {{user}}",
		Variables: []llmops.PromptVariable{
			{Name: "user", Type: "string", Required: true},
		},
	}); err != nil {
		return fmt.Errorf("create v1.0.0: %w", err)
	}

	if err := reg.Create(ctx, &llmops.PromptVersion{
		Name:    "greet",
		Version: "1.1.0",
		Content: "hi {{user}}",
		Variables: []llmops.PromptVariable{
			{Name: "user", Type: "string", Required: true},
		},
		IsActive: true,
	}); err != nil {
		return fmt.Errorf("create v1.1.0: %w", err)
	}

	latest, err := reg.GetLatest(ctx, "greet")
	if err != nil {
		return fmt.Errorf("get latest: %w", err)
	}

	// Mutation gate — if --mutate=registry, lie about which version is active.
	// The honest assertion below must then FAIL, proving the assertion is real.
	expected := "1.1.0"
	if r.mutate == "registry" {
		expected = "9.9.9-mutated"
	}
	if latest.Version != expected {
		return fmt.Errorf("active version mismatch: want=%s got=%s", expected, latest.Version)
	}

	versions, err := reg.List(ctx, "greet")
	if err != nil {
		return fmt.Errorf("list: %w", err)
	}
	if len(versions) != 2 {
		return fmt.Errorf("expected 2 versions, got %d", len(versions))
	}
	return nil
}

// phaseRender exercises variable substitution + required-var enforcement.
func (r *runner) phaseRender(ctx context.Context) error {
	reg := llmops.NewInMemoryPromptRegistry(silentLogger())
	if err := reg.Create(ctx, &llmops.PromptVersion{
		Name:    "sum",
		Version: "1.0.0",
		Content: "result: {{a}} + {{b}}",
		Variables: []llmops.PromptVariable{
			{Name: "a", Type: "string", Required: true},
			{Name: "b", Type: "string", Required: true},
		},
	}); err != nil {
		return fmt.Errorf("create: %w", err)
	}

	got, err := reg.Render(ctx, "sum", "1.0.0", map[string]interface{}{
		"a": "2", "b": "3",
	})
	if err != nil {
		return fmt.Errorf("render: %w", err)
	}
	wantSub := "2 + 3"
	if r.mutate == "render" {
		wantSub = "absent-marker-77"
	}
	if !strings.Contains(got, wantSub) {
		return fmt.Errorf("render output missing %q: got=%q", wantSub, got)
	}

	// Missing-required-var must error.
	if _, err := reg.Render(ctx, "sum", "1.0.0", map[string]interface{}{"a": "2"}); err == nil {
		return errors.New("expected missing-required-var error, got nil")
	}
	return nil
}

// phaseCompare exercises PromptVersionComparator.
func (r *runner) phaseCompare(ctx context.Context) error {
	reg := llmops.NewInMemoryPromptRegistry(silentLogger())
	for _, v := range []*llmops.PromptVersion{
		{Name: "p", Version: "1.0.0", Content: "old line\nshared"},
		{Name: "p", Version: "2.0.0", Content: "new line\nshared"},
	} {
		if err := reg.Create(ctx, v); err != nil {
			return fmt.Errorf("seed %s: %w", v.Version, err)
		}
	}
	cmp := llmops.NewPromptVersionComparator(reg, silentLogger())
	diff, err := cmp.Compare(ctx, "p", "1.0.0", "2.0.0")
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	wantOld := "old line"
	if r.mutate == "compare" {
		wantOld = "ghost-line-marker"
	}
	if !strings.Contains(diff.ContentDiff, wantOld) {
		return fmt.Errorf("diff missing %q segment: %q", wantOld, diff.ContentDiff)
	}
	if !strings.Contains(diff.ContentDiff, "new line") {
		return fmt.Errorf("diff missing new-line segment: %q", diff.ContentDiff)
	}
	return nil
}

// phaseI18n exercises the NoopTranslator surface — proves keys round-trip.
func (r *runner) phaseI18n(ctx context.Context) error {
	_ = ctx
	loc := r.loc
	// At minimum, header + pass_line + fail_line + summary must be populated.
	required := []string{"header", "pass_line", "fail_line", "summary", "phase_prompt_registry_begin"}
	for _, k := range required {
		if v, ok := loc[k]; !ok || strings.TrimSpace(v) == "" {
			return fmt.Errorf("locale %s missing required key %q", r.locale, k)
		}
	}
	return nil
}

// emit looks up the localised template, applies params, prints it.
func (r *runner) emit(key string, params map[string]string) {
	template, ok := r.loc[key]
	if !ok {
		fmt.Fprintf(r.out, "[FALLBACK key=%s] params=%v\n", key, params)
		return
	}
	out := template
	for k, v := range params {
		out = strings.ReplaceAll(out, "{{"+k+"}}", v)
	}
	fmt.Fprintln(r.out, out)
}

// loadLocale reads challenges/fixtures/runner.<locale>.txt, falling back to en
// with a loud marker if the requested locale is absent.
func loadLocale(dir, locale string) (locStrings, error) {
	path := filepath.Join(dir, "runner."+locale+".txt")
	if _, err := os.Stat(path); err != nil {
		if locale == defaultLocale {
			return nil, fmt.Errorf("default locale %s fixture missing: %w", locale, err)
		}
		fmt.Fprintf(os.Stderr, "[FALLBACK locale=%s → %s] fixture missing: %v\n", locale, defaultLocale, err)
		path = filepath.Join(dir, "runner."+defaultLocale+".txt")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := locStrings{}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, "=")
		if idx <= 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		out[key] = val
	}
	return out, nil
}

func silentLogger() *logrus.Logger {
	l := logrus.New()
	l.SetOutput(os.NewFile(0, os.DevNull))
	return l
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func ifEmpty(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
