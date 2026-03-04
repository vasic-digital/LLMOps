package llmops

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

// --- Helpers ---

func newTestPromptRegistry() *InMemoryPromptRegistry {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	return NewInMemoryPromptRegistry(logger)
}

func createTestPrompt(name, version, content string) *PromptVersion {
	return &PromptVersion{
		Name:    name,
		Version: version,
		Content: content,
	}
}

// --- Constructor tests ---

func TestNewInMemoryPromptRegistry_NilLogger(t *testing.T) {
	r := NewInMemoryPromptRegistry(nil)
	require.NotNil(t, r)
	require.NotNil(t, r.logger)
}

func TestNewInMemoryPromptRegistry_WithLogger(t *testing.T) {
	logger := logrus.New()
	r := NewInMemoryPromptRegistry(logger)
	require.NotNil(t, r)
	assert.Equal(t, logger, r.logger)
}

// --- Create tests ---

func TestInMemoryPromptRegistry_Create_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := createTestPrompt("greeting", "1.0.0", "Hello, {{name}}!")
	err := r.Create(ctx, p)
	require.NoError(t, err)
	assert.NotEmpty(t, p.ID)
	assert.True(t, p.IsActive, "first version should be active")
	assert.False(t, p.CreatedAt.IsZero())
	assert.False(t, p.UpdatedAt.IsZero())
}

func TestInMemoryPromptRegistry_Create_WithExplicitID(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{ID: "custom-id", Name: "greeting", Version: "1.0.0", Content: "Hello!"}
	err := r.Create(ctx, p)
	require.NoError(t, err)
	assert.Equal(t, "custom-id", p.ID)
}

func TestInMemoryPromptRegistry_Create_EmptyName(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{Version: "1.0.0", Content: "Hello!"}
	err := r.Create(ctx, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt name is required")
}

func TestInMemoryPromptRegistry_Create_EmptyVersion(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{Name: "greeting", Content: "Hello!"}
	err := r.Create(ctx, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt version is required")
}

func TestInMemoryPromptRegistry_Create_EmptyContent(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{Name: "greeting", Version: "1.0.0"}
	err := r.Create(ctx, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt content is required")
}

func TestInMemoryPromptRegistry_Create_DuplicateVersion(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	err := r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hi!"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt version already exists")
}

func TestInMemoryPromptRegistry_Create_FirstVersionIsActive(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := createTestPrompt("greeting", "1.0.0", "Hello!")
	require.NoError(t, r.Create(ctx, p))
	assert.True(t, p.IsActive)
}

func TestInMemoryPromptRegistry_Create_SecondVersionNotActive(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	p2 := createTestPrompt("greeting", "2.0.0", "Hi!")
	require.NoError(t, r.Create(ctx, p2))
	assert.False(t, p2.IsActive, "second version should NOT be active unless explicitly set")
}

func TestInMemoryPromptRegistry_Create_SecondVersionExplicitlyActive(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	p2 := &PromptVersion{Name: "greeting", Version: "2.0.0", Content: "Hi!", IsActive: true}
	require.NoError(t, r.Create(ctx, p2))
	assert.True(t, p2.IsActive)
}

func TestInMemoryPromptRegistry_Create_WithVariables(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! Your age is {{age}}.",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "age", Type: "int", Required: false, Default: 25},
		},
	}
	err := r.Create(ctx, p)
	require.NoError(t, err)
}

func TestInMemoryPromptRegistry_Create_UndefinedVariableWarning(t *testing.T) {
	// This should not error — just warn
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! {{undefined_var}}",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}
	err := r.Create(ctx, p)
	require.NoError(t, err, "undefined variables should warn, not error")
}

func TestInMemoryPromptRegistry_Create_DifferentPromptNames(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("farewell", "1.0.0", "Goodbye!")))

	g, err := r.Get(ctx, "greeting", "1.0.0")
	require.NoError(t, err)
	assert.Equal(t, "Hello!", g.Content)

	f, err := r.Get(ctx, "farewell", "1.0.0")
	require.NoError(t, err)
	assert.Equal(t, "Goodbye!", f.Content)
}

// --- Get tests ---

func TestInMemoryPromptRegistry_Get_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	got, err := r.Get(ctx, "greeting", "1.0.0")
	require.NoError(t, err)
	assert.Equal(t, "greeting", got.Name)
	assert.Equal(t, "1.0.0", got.Version)
	assert.Equal(t, "Hello!", got.Content)
}

func TestInMemoryPromptRegistry_Get_PromptNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	_, err := r.Get(ctx, "nonexistent", "1.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt not found")
}

func TestInMemoryPromptRegistry_Get_VersionNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	_, err := r.Get(ctx, "greeting", "2.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "version not found")
}

// --- GetLatest tests ---

func TestInMemoryPromptRegistry_GetLatest_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	got, err := r.GetLatest(ctx, "greeting")
	require.NoError(t, err)
	assert.Equal(t, "1.0.0", got.Version)
}

func TestInMemoryPromptRegistry_GetLatest_AfterActivation(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))
	require.NoError(t, r.Activate(ctx, "greeting", "2.0.0"))

	got, err := r.GetLatest(ctx, "greeting")
	require.NoError(t, err)
	assert.Equal(t, "2.0.0", got.Version)
}

func TestInMemoryPromptRegistry_GetLatest_NotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	_, err := r.GetLatest(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no active version")
}

// --- List tests ---

func TestInMemoryPromptRegistry_List_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	time.Sleep(time.Millisecond)
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))

	versions, err := r.List(ctx, "greeting")
	require.NoError(t, err)
	assert.Len(t, versions, 2)
	// Newest first
	assert.Equal(t, "2.0.0", versions[0].Version)
	assert.Equal(t, "1.0.0", versions[1].Version)
}

func TestInMemoryPromptRegistry_List_NotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	versions, err := r.List(ctx, "nonexistent")
	require.NoError(t, err)
	assert.Empty(t, versions)
}

func TestInMemoryPromptRegistry_List_EmptyResult(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	versions, err := r.List(ctx, "nonexistent")
	require.NoError(t, err)
	assert.NotNil(t, versions)
	assert.Len(t, versions, 0)
}

// --- ListAll tests ---

func TestInMemoryPromptRegistry_ListAll_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("farewell", "1.0.0", "Goodbye!")))

	all, err := r.ListAll(ctx)
	require.NoError(t, err)
	assert.Len(t, all, 3)
	// Sorted by name then newest first
	assert.Equal(t, "farewell", all[0].Name)
	assert.Equal(t, "greeting", all[1].Name)
}

func TestInMemoryPromptRegistry_ListAll_Empty(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	all, err := r.ListAll(ctx)
	require.NoError(t, err)
	assert.Empty(t, all)
}

// --- Activate tests ---

func TestInMemoryPromptRegistry_Activate_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))

	err := r.Activate(ctx, "greeting", "2.0.0")
	require.NoError(t, err)

	// Version 2.0.0 should be active
	v2, _ := r.Get(ctx, "greeting", "2.0.0")
	assert.True(t, v2.IsActive)

	// Version 1.0.0 should be deactivated
	v1, _ := r.Get(ctx, "greeting", "1.0.0")
	assert.False(t, v1.IsActive)
}

func TestInMemoryPromptRegistry_Activate_SameVersion(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	err := r.Activate(ctx, "greeting", "1.0.0")
	require.NoError(t, err)

	v, _ := r.Get(ctx, "greeting", "1.0.0")
	assert.True(t, v.IsActive)
}

func TestInMemoryPromptRegistry_Activate_PromptNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	err := r.Activate(ctx, "nonexistent", "1.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt not found")
}

func TestInMemoryPromptRegistry_Activate_VersionNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	err := r.Activate(ctx, "greeting", "3.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "version not found")
}

func TestInMemoryPromptRegistry_Activate_UpdatesTimestamp(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := createTestPrompt("greeting", "1.0.0", "Hello!")
	require.NoError(t, r.Create(ctx, p))
	originalUpdated := p.UpdatedAt

	time.Sleep(time.Millisecond)
	require.NoError(t, r.Activate(ctx, "greeting", "1.0.0"))

	got, _ := r.Get(ctx, "greeting", "1.0.0")
	assert.True(t, got.UpdatedAt.After(originalUpdated))
}

// --- Delete tests ---

func TestInMemoryPromptRegistry_Delete_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))

	// 1.0.0 is active, so delete 2.0.0
	err := r.Delete(ctx, "greeting", "2.0.0")
	require.NoError(t, err)

	_, err = r.Get(ctx, "greeting", "2.0.0")
	assert.Error(t, err)

	versions, _ := r.List(ctx, "greeting")
	assert.Len(t, versions, 1)
}

func TestInMemoryPromptRegistry_Delete_ActiveVersion(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	err := r.Delete(ctx, "greeting", "1.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot delete active version")
}

func TestInMemoryPromptRegistry_Delete_PromptNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	err := r.Delete(ctx, "nonexistent", "1.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt not found")
}

func TestInMemoryPromptRegistry_Delete_VersionNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	err := r.Delete(ctx, "greeting", "2.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "version not found")
}

func TestInMemoryPromptRegistry_Delete_LastInactiveVersion_CleansUp(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))

	// Activate 2.0.0, then delete 1.0.0
	require.NoError(t, r.Activate(ctx, "greeting", "2.0.0"))
	require.NoError(t, r.Delete(ctx, "greeting", "1.0.0"))

	versions, _ := r.List(ctx, "greeting")
	assert.Len(t, versions, 1)

	// Now delete the active version is not allowed
	err := r.Delete(ctx, "greeting", "2.0.0")
	assert.Error(t, err)
}

func TestInMemoryPromptRegistry_Delete_AllInactiveVersions_CleansUpPrompt(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hi!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "3.0.0", "Hey!")))

	// Activate 2.0.0, delete 1.0.0 and 3.0.0
	require.NoError(t, r.Activate(ctx, "greeting", "2.0.0"))
	require.NoError(t, r.Delete(ctx, "greeting", "1.0.0"))
	require.NoError(t, r.Delete(ctx, "greeting", "3.0.0"))

	versions, _ := r.List(ctx, "greeting")
	assert.Len(t, versions, 1)
	assert.Equal(t, "2.0.0", versions[0].Version)
}

// --- Render tests ---

func TestInMemoryPromptRegistry_Render_SimpleSubstitution(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))

	rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{"name": "Alice"})
	require.NoError(t, err)
	assert.Equal(t, "Hello, Alice!", rendered)
}

func TestInMemoryPromptRegistry_Render_MultipleVariables(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! You are {{age}} years old.",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "age", Type: "int", Required: true},
		},
	}))

	rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{
		"name": "Bob",
		"age":  30,
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello, Bob! You are 30 years old.", rendered)
}

func TestInMemoryPromptRegistry_Render_DefaultValue(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! Role: {{role}}",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "role", Type: "string", Required: false, Default: "user"},
		},
	}))

	rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{"name": "Alice"})
	require.NoError(t, err)
	assert.Equal(t, "Hello, Alice! Role: user", rendered)
}

func TestInMemoryPromptRegistry_Render_MissingRequiredVariable(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))

	_, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing required variable: name")
}

func TestInMemoryPromptRegistry_Render_OptionalVariableMissing(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! {{optional}}",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "optional", Type: "string", Required: false},
		},
	}))

	rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{"name": "Alice"})
	require.NoError(t, err)
	// Optional variable without default is skipped — placeholder remains
	assert.Equal(t, "Hello, Alice! {{optional}}", rendered)
}

func TestInMemoryPromptRegistry_Render_Validation_Passes(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "email",
		Version: "1.0.0",
		Content: "Send to {{email}}",
		Variables: []PromptVariable{
			{Name: "email", Type: "string", Required: true, Validation: `^.+@.+\..+$`},
		},
	}))

	rendered, err := r.Render(ctx, "email", "1.0.0", map[string]interface{}{"email": "test@example.com"})
	require.NoError(t, err)
	assert.Equal(t, "Send to test@example.com", rendered)
}

func TestInMemoryPromptRegistry_Render_Validation_Fails(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "email",
		Version: "1.0.0",
		Content: "Send to {{email}}",
		Variables: []PromptVariable{
			{Name: "email", Type: "string", Required: true, Validation: `^.+@.+\..+$`},
		},
	}))

	_, err := r.Render(ctx, "email", "1.0.0", map[string]interface{}{"email": "not-an-email"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed validation")
}

func TestInMemoryPromptRegistry_Render_ExtraVariables(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! {{extra}}",
	}))

	// Extra variables not in the Variables list should still be substituted
	rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{
		"name":  "Alice",
		"extra": "Welcome!",
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello, Alice! Welcome!", rendered)
}

func TestInMemoryPromptRegistry_Render_PromptNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	_, err := r.Render(ctx, "nonexistent", "1.0.0", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt not found")
}

func TestInMemoryPromptRegistry_Render_VersionNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	_, err := r.Render(ctx, "greeting", "2.0.0", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "version not found")
}

func TestInMemoryPromptRegistry_Render_NilVars(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	rendered, err := r.Render(ctx, "greeting", "1.0.0", nil)
	require.NoError(t, err)
	assert.Equal(t, "Hello!", rendered)
}

func TestInMemoryPromptRegistry_Render_EmptyVars(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello, {{name}}!")))

	rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{})
	require.NoError(t, err)
	assert.Equal(t, "Hello, {{name}}!", rendered)
}

func TestInMemoryPromptRegistry_Render_RepeatedVariable(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "echo",
		Version: "1.0.0",
		Content: "{{word}} and {{word}} again",
		Variables: []PromptVariable{
			{Name: "word", Type: "string", Required: true},
		},
	}))

	rendered, err := r.Render(ctx, "echo", "1.0.0", map[string]interface{}{"word": "hello"})
	require.NoError(t, err)
	assert.Equal(t, "hello and hello again", rendered)
}

// --- Concurrent access tests ---

func TestInMemoryPromptRegistry_ConcurrentCreates(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	var wg sync.WaitGroup
	n := 20
	errs := make([]error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			p := createTestPrompt(
				fmt.Sprintf("prompt-%d", idx),
				"1.0.0",
				fmt.Sprintf("Content %d", idx),
			)
			errs[idx] = r.Create(ctx, p)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoError(t, err, "prompt %d", i)
	}

	all, _ := r.ListAll(ctx)
	assert.Len(t, all, n)
}

func TestInMemoryPromptRegistry_ConcurrentReadWrite(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	// Seed with some prompts
	for i := 0; i < 5; i++ {
		require.NoError(t, r.Create(ctx, createTestPrompt(
			fmt.Sprintf("prompt-%d", i), "1.0.0", fmt.Sprintf("Content %d", i),
		)))
	}

	var wg sync.WaitGroup

	// Concurrent readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = r.ListAll(ctx)
			_, _ = r.Get(ctx, "prompt-0", "1.0.0")
			_, _ = r.GetLatest(ctx, "prompt-0")
			_, _ = r.List(ctx, "prompt-0")
		}()
	}

	// Concurrent writers
	for i := 5; i < 15; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_ = r.Create(ctx, createTestPrompt(
				fmt.Sprintf("prompt-%d", idx), "1.0.0", fmt.Sprintf("Content %d", idx),
			))
		}(i)
	}

	wg.Wait()
}

func TestInMemoryPromptRegistry_ConcurrentRender(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			rendered, err := r.Render(ctx, "greeting", "1.0.0", map[string]interface{}{
				"name": fmt.Sprintf("User-%d", idx),
			})
			assert.NoError(t, err)
			assert.Contains(t, rendered, fmt.Sprintf("User-%d", idx))
		}(i)
	}
	wg.Wait()
}

// --- PromptVersionComparator tests ---

func TestNewPromptVersionComparator(t *testing.T) {
	r := newTestPromptRegistry()
	logger := logrus.New()
	comp := NewPromptVersionComparator(r, logger)
	require.NotNil(t, comp)
	assert.Equal(t, r, comp.registry)
	assert.Equal(t, logger, comp.logger)
}

func TestPromptVersionComparator_Compare_Success(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))
	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "2.0.0",
		Content: "Hi, {{name}}! Welcome, {{role}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "role", Type: "string", Required: false},
		},
	}))

	comp := NewPromptVersionComparator(r, nil)
	diff, err := comp.Compare(ctx, "greeting", "1.0.0", "2.0.0")
	require.NoError(t, err)
	assert.Equal(t, "1.0.0", diff.OldVersion)
	assert.Equal(t, "2.0.0", diff.NewVersion)
	assert.NotEmpty(t, diff.ContentDiff)
	assert.Contains(t, diff.AddedVars, "role")
	assert.Empty(t, diff.RemovedVars)
}

func TestPromptVersionComparator_Compare_RemovedVariables(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}! Role: {{role}}",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "role", Type: "string", Required: false},
		},
	}))
	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "2.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))

	comp := NewPromptVersionComparator(r, nil)
	diff, err := comp.Compare(ctx, "greeting", "1.0.0", "2.0.0")
	require.NoError(t, err)
	assert.Contains(t, diff.RemovedVars, "role")
	assert.Empty(t, diff.AddedVars)
}

func TestPromptVersionComparator_Compare_ChangedVariables(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "1.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))
	require.NoError(t, r.Create(ctx, &PromptVersion{
		Name:    "greeting",
		Version: "2.0.0",
		Content: "Hello, {{name}}!",
		Variables: []PromptVariable{
			{Name: "name", Type: "int", Required: false}, // changed type and required
		},
	}))

	comp := NewPromptVersionComparator(r, nil)
	diff, err := comp.Compare(ctx, "greeting", "1.0.0", "2.0.0")
	require.NoError(t, err)
	assert.Contains(t, diff.ChangedVars, "name")
}

func TestPromptVersionComparator_Compare_IdenticalVersions(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))
	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hello!")))

	comp := NewPromptVersionComparator(r, nil)
	diff, err := comp.Compare(ctx, "greeting", "1.0.0", "2.0.0")
	require.NoError(t, err)
	assert.Empty(t, diff.AddedVars)
	assert.Empty(t, diff.RemovedVars)
	assert.Empty(t, diff.ChangedVars)
}

func TestPromptVersionComparator_Compare_Version1NotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "2.0.0", "Hello!")))

	comp := NewPromptVersionComparator(r, nil)
	_, err := comp.Compare(ctx, "greeting", "1.0.0", "2.0.0")
	require.Error(t, err)
}

func TestPromptVersionComparator_Compare_Version2NotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	require.NoError(t, r.Create(ctx, createTestPrompt("greeting", "1.0.0", "Hello!")))

	comp := NewPromptVersionComparator(r, nil)
	_, err := comp.Compare(ctx, "greeting", "1.0.0", "2.0.0")
	require.Error(t, err)
}

func TestPromptVersionComparator_Compare_PromptNotFound(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	comp := NewPromptVersionComparator(r, nil)
	_, err := comp.Compare(ctx, "nonexistent", "1.0.0", "2.0.0")
	require.Error(t, err)
}

// --- computeDiff tests ---

func TestPromptVersionComparator_ComputeDiff_IdenticalContent(t *testing.T) {
	comp := &PromptVersionComparator{}
	diff := comp.computeDiff("Hello\nWorld", "Hello\nWorld")
	assert.Contains(t, diff, "  Hello")
	assert.Contains(t, diff, "  World")
	assert.NotContains(t, diff, "+ ")
	assert.NotContains(t, diff, "- ")
}

func TestPromptVersionComparator_ComputeDiff_AddedLines(t *testing.T) {
	comp := &PromptVersionComparator{}
	diff := comp.computeDiff("Hello", "Hello\nWorld")
	assert.Contains(t, diff, "  Hello")
	assert.Contains(t, diff, "+ World")
}

func TestPromptVersionComparator_ComputeDiff_RemovedLines(t *testing.T) {
	comp := &PromptVersionComparator{}
	diff := comp.computeDiff("Hello\nWorld", "Hello")
	assert.Contains(t, diff, "  Hello")
	assert.Contains(t, diff, "- World")
}

func TestPromptVersionComparator_ComputeDiff_ChangedLines(t *testing.T) {
	comp := &PromptVersionComparator{}
	diff := comp.computeDiff("Hello", "Hi")
	assert.Contains(t, diff, "- Hello")
	assert.Contains(t, diff, "+ Hi")
}

func TestPromptVersionComparator_ComputeDiff_EmptyOld(t *testing.T) {
	comp := &PromptVersionComparator{}
	diff := comp.computeDiff("", "Hello")
	// Empty splits into [""], so we get a change from "" to "Hello"
	assert.Contains(t, diff, "+ Hello")
}

func TestPromptVersionComparator_ComputeDiff_EmptyNew(t *testing.T) {
	comp := &PromptVersionComparator{}
	diff := comp.computeDiff("Hello", "")
	assert.Contains(t, diff, "- Hello")
}

func TestPromptVersionComparator_ComputeDiff_MultipleChanges(t *testing.T) {
	comp := &PromptVersionComparator{}
	old := "Line 1\nLine 2\nLine 3"
	newContent := "Line 1\nModified Line 2\nLine 3\nLine 4"
	diff := comp.computeDiff(old, newContent)
	assert.Contains(t, diff, "  Line 1")
	assert.Contains(t, diff, "- Line 2")
	assert.Contains(t, diff, "+ Modified Line 2")
}

// --- Full lifecycle test ---

func TestInMemoryPromptRegistry_FullLifecycle(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	// 1. Create first version
	v1 := &PromptVersion{
		Name:    "qa-prompt",
		Version: "1.0.0",
		Content: "Answer the question: {{question}}",
		Variables: []PromptVariable{
			{Name: "question", Type: "string", Required: true},
		},
		Author:      "dev-team",
		Description: "Q&A prompt",
		Tags:        []string{"qa", "v1"},
	}
	require.NoError(t, r.Create(ctx, v1))
	assert.True(t, v1.IsActive)

	// 2. Get latest
	latest, err := r.GetLatest(ctx, "qa-prompt")
	require.NoError(t, err)
	assert.Equal(t, "1.0.0", latest.Version)

	// 3. Create second version
	v2 := &PromptVersion{
		Name:    "qa-prompt",
		Version: "2.0.0",
		Content: "You are a helpful assistant. Answer: {{question}} Context: {{context}}",
		Variables: []PromptVariable{
			{Name: "question", Type: "string", Required: true},
			{Name: "context", Type: "string", Required: false, Default: "general"},
		},
		Author: "dev-team",
		Tags:   []string{"qa", "v2"},
	}
	require.NoError(t, r.Create(ctx, v2))

	// 4. List all versions
	versions, err := r.List(ctx, "qa-prompt")
	require.NoError(t, err)
	assert.Len(t, versions, 2)

	// 5. Activate v2
	require.NoError(t, r.Activate(ctx, "qa-prompt", "2.0.0"))

	// 6. Render v2
	rendered, err := r.Render(ctx, "qa-prompt", "2.0.0", map[string]interface{}{
		"question": "What is Go?",
	})
	require.NoError(t, err)
	assert.Equal(t, "You are a helpful assistant. Answer: What is Go? Context: general", rendered)

	// 7. Compare versions
	comp := NewPromptVersionComparator(r, nil)
	diff, err := comp.Compare(ctx, "qa-prompt", "1.0.0", "2.0.0")
	require.NoError(t, err)
	assert.Contains(t, diff.AddedVars, "context")
	assert.NotEmpty(t, diff.ContentDiff)

	// 8. Delete v1 (no longer active)
	require.NoError(t, r.Delete(ctx, "qa-prompt", "1.0.0"))
	versions, _ = r.List(ctx, "qa-prompt")
	assert.Len(t, versions, 1)
}

// --- Interface compliance ---

func TestInMemoryPromptRegistry_ImplementsPromptRegistry(t *testing.T) {
	var _ PromptRegistry = (*InMemoryPromptRegistry)(nil)
}

// --- Edge case: metadata and optional fields ---

func TestInMemoryPromptRegistry_Create_WithAllOptionalFields(t *testing.T) {
	r := newTestPromptRegistry()
	ctx := context.Background()

	p := &PromptVersion{
		Name:        "full-prompt",
		Version:     "1.0.0",
		Content:     "Hello!",
		Author:      "test-author",
		Description: "A test prompt",
		Tags:        []string{"tag1", "tag2"},
		Metadata:    map[string]interface{}{"key": "value", "count": 42},
	}
	err := r.Create(ctx, p)
	require.NoError(t, err)

	got, _ := r.Get(ctx, "full-prompt", "1.0.0")
	assert.Equal(t, "test-author", got.Author)
	assert.Equal(t, "A test prompt", got.Description)
	assert.Equal(t, []string{"tag1", "tag2"}, got.Tags)
	assert.Equal(t, "value", got.Metadata["key"])
}
