package llmops

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// InMemoryPromptRegistry implements PromptRegistry with in-memory storage
type InMemoryPromptRegistry struct {
	prompts map[string]map[string]*PromptVersion // name -> version -> prompt
	active  map[string]string                    // name -> active version
	mu      sync.RWMutex
	logger  *logrus.Logger
}

// NewInMemoryPromptRegistry creates a new in-memory prompt registry
func NewInMemoryPromptRegistry(logger *logrus.Logger) *InMemoryPromptRegistry {
	if logger == nil {
		logger = logrus.New()
	}
	return &InMemoryPromptRegistry{
		prompts: make(map[string]map[string]*PromptVersion),
		active:  make(map[string]string),
		logger:  logger,
	}
}

// Create creates a new prompt version
func (r *InMemoryPromptRegistry) Create(ctx context.Context, prompt *PromptVersion) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if prompt.Name == "" {
		return fmt.Errorf("prompt name is required")
	}
	if prompt.Version == "" {
		return fmt.Errorf("prompt version is required")
	}
	if prompt.Content == "" {
		return fmt.Errorf("prompt content is required")
	}

	// Generate ID if not present
	if prompt.ID == "" {
		prompt.ID = uuid.New().String()
	}

	// Initialize maps
	if r.prompts[prompt.Name] == nil {
		r.prompts[prompt.Name] = make(map[string]*PromptVersion)
	}

	// Check if version exists
	if _, exists := r.prompts[prompt.Name][prompt.Version]; exists {
		return fmt.Errorf("prompt version already exists: %s/%s", prompt.Name, prompt.Version)
	}

	// Validate variables in content
	if err := r.validateVariables(prompt); err != nil {
		return err
	}

	prompt.CreatedAt = time.Now()
	prompt.UpdatedAt = time.Now()

	r.prompts[prompt.Name][prompt.Version] = prompt

	// Set as active if first version or explicitly active
	if len(r.prompts[prompt.Name]) == 1 || prompt.IsActive {
		r.active[prompt.Name] = prompt.Version
		prompt.IsActive = true
	}

	r.logger.WithFields(logrus.Fields{
		"name":    prompt.Name,
		"version": prompt.Version,
		"id":      prompt.ID,
	}).Debug("Prompt version created")

	return nil
}

// Get retrieves a specific version
func (r *InMemoryPromptRegistry) Get(ctx context.Context, name, version string) (*PromptVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, ok := r.prompts[name]
	if !ok {
		return nil, fmt.Errorf("prompt not found: %s", name)
	}

	prompt, ok := versions[version]
	if !ok {
		return nil, fmt.Errorf("version not found: %s/%s", name, version)
	}

	return prompt, nil
}

// GetLatest retrieves the latest active version
func (r *InMemoryPromptRegistry) GetLatest(ctx context.Context, name string) (*PromptVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	activeVersion, ok := r.active[name]
	if !ok {
		return nil, fmt.Errorf("no active version for prompt: %s", name)
	}

	return r.prompts[name][activeVersion], nil
}

// List lists all versions of a prompt
func (r *InMemoryPromptRegistry) List(ctx context.Context, name string) ([]*PromptVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, ok := r.prompts[name]
	if !ok {
		return []*PromptVersion{}, nil
	}

	result := make([]*PromptVersion, 0, len(versions))
	for _, v := range versions {
		result = append(result, v)
	}

	// Sort by creation time (newest first)
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})

	return result, nil
}

// ListAll lists all prompts
func (r *InMemoryPromptRegistry) ListAll(ctx context.Context) ([]*PromptVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var result []*PromptVersion
	for _, versions := range r.prompts {
		for _, v := range versions {
			result = append(result, v)
		}
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name ||
			(result[i].Name == result[j].Name && result[i].CreatedAt.After(result[j].CreatedAt))
	})

	return result, nil
}

// Activate sets a version as active
func (r *InMemoryPromptRegistry) Activate(ctx context.Context, name, version string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions, ok := r.prompts[name]
	if !ok {
		return fmt.Errorf("prompt not found: %s", name)
	}

	prompt, ok := versions[version]
	if !ok {
		return fmt.Errorf("version not found: %s/%s", name, version)
	}

	// Deactivate current active
	if currentActive, exists := r.active[name]; exists {
		if v, ok := versions[currentActive]; ok {
			v.IsActive = false
		}
	}

	// Activate new version
	r.active[name] = version
	prompt.IsActive = true
	prompt.UpdatedAt = time.Now()

	r.logger.WithFields(logrus.Fields{
		"name":    name,
		"version": version,
	}).Info("Prompt version activated")

	return nil
}

// Delete removes a prompt version
func (r *InMemoryPromptRegistry) Delete(ctx context.Context, name, version string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions, ok := r.prompts[name]
	if !ok {
		return fmt.Errorf("prompt not found: %s", name)
	}

	if _, ok := versions[version]; !ok {
		return fmt.Errorf("version not found: %s/%s", name, version)
	}

	// Can't delete active version
	if r.active[name] == version {
		return fmt.Errorf("cannot delete active version: %s/%s", name, version)
	}

	delete(versions, version)

	// Clean up if no versions left
	if len(versions) == 0 {
		delete(r.prompts, name)
		delete(r.active, name)
	}

	r.logger.WithFields(logrus.Fields{
		"name":    name,
		"version": version,
	}).Info("Prompt version deleted")

	return nil
}

// Render renders a prompt with variables
func (r *InMemoryPromptRegistry) Render(ctx context.Context, name, version string, vars map[string]interface{}) (string, error) {
	prompt, err := r.Get(ctx, name, version)
	if err != nil {
		return "", err
	}

	rendered := prompt.Content

	// Check required variables
	for _, v := range prompt.Variables {
		value, ok := vars[v.Name]
		if !ok {
			if v.Default != nil {
				value = v.Default
			} else if v.Required {
				return "", fmt.Errorf("missing required variable: %s", v.Name)
			} else {
				continue
			}
		}

		// Validate value
		if v.Validation != "" {
			if matched, _ := regexp.MatchString(v.Validation, fmt.Sprintf("%v", value)); !matched {
				return "", fmt.Errorf("variable %s failed validation", v.Name)
			}
		}

		// Replace placeholder
		placeholder := fmt.Sprintf("{{%s}}", v.Name)
		rendered = strings.ReplaceAll(rendered, placeholder, fmt.Sprintf("%v", value))
	}

	// Replace any remaining variables from vars
	for k, v := range vars {
		placeholder := fmt.Sprintf("{{%s}}", k)
		rendered = strings.ReplaceAll(rendered, placeholder, fmt.Sprintf("%v", v))
	}

	return rendered, nil
}

func (r *InMemoryPromptRegistry) validateVariables(prompt *PromptVersion) error {
	// Find all placeholders in content
	re := regexp.MustCompile(`\{\{(\w+)\}\}`)
	matches := re.FindAllStringSubmatch(prompt.Content, -1)

	definedVars := make(map[string]bool)
	for _, v := range prompt.Variables {
		definedVars[v.Name] = true
	}

	// Warn about undefined variables (but don't fail)
	for _, match := range matches {
		if len(match) > 1 {
			varName := match[1]
			if !definedVars[varName] {
				r.logger.WithFields(logrus.Fields{
					"prompt":   prompt.Name,
					"variable": varName,
				}).Warn("Undefined variable in prompt")
			}
		}
	}

	return nil
}

// PromptVersionComparator compares prompt versions
type PromptVersionComparator struct {
	registry PromptRegistry
	logger   *logrus.Logger
}

// NewPromptVersionComparator creates a new comparator
func NewPromptVersionComparator(registry PromptRegistry, logger *logrus.Logger) *PromptVersionComparator {
	return &PromptVersionComparator{
		registry: registry,
		logger:   logger,
	}
}

// PromptDiff represents differences between versions
type PromptDiff struct {
	OldVersion  string   `json:"old_version"`
	NewVersion  string   `json:"new_version"`
	ContentDiff string   `json:"content_diff"`
	AddedVars   []string `json:"added_vars,omitempty"`
	RemovedVars []string `json:"removed_vars,omitempty"`
	ChangedVars []string `json:"changed_vars,omitempty"`
}

// Compare compares two prompt versions
func (c *PromptVersionComparator) Compare(ctx context.Context, name, version1, version2 string) (*PromptDiff, error) {
	p1, err := c.registry.Get(ctx, name, version1)
	if err != nil {
		return nil, err
	}

	p2, err := c.registry.Get(ctx, name, version2)
	if err != nil {
		return nil, err
	}

	diff := &PromptDiff{
		OldVersion: version1,
		NewVersion: version2,
	}

	// Simple content diff (line-based)
	diff.ContentDiff = c.computeDiff(p1.Content, p2.Content)

	// Compare variables
	v1Vars := make(map[string]PromptVariable)
	for _, v := range p1.Variables {
		v1Vars[v.Name] = v
	}

	v2Vars := make(map[string]PromptVariable)
	for _, v := range p2.Variables {
		v2Vars[v.Name] = v
	}

	for name := range v2Vars {
		if _, exists := v1Vars[name]; !exists {
			diff.AddedVars = append(diff.AddedVars, name)
		}
	}

	for name := range v1Vars {
		if _, exists := v2Vars[name]; !exists {
			diff.RemovedVars = append(diff.RemovedVars, name)
		}
	}

	for name, v1 := range v1Vars {
		if v2, exists := v2Vars[name]; exists {
			if v1.Type != v2.Type || v1.Required != v2.Required {
				diff.ChangedVars = append(diff.ChangedVars, name)
			}
		}
	}

	return diff, nil
}

func (c *PromptVersionComparator) computeDiff(old, new string) string {
	oldLines := strings.Split(old, "\n")
	newLines := strings.Split(new, "\n")

	var diff strings.Builder
	i, j := 0, 0

	for i < len(oldLines) || j < len(newLines) {
		if i >= len(oldLines) {
			diff.WriteString(fmt.Sprintf("+ %s\n", newLines[j]))
			j++
		} else if j >= len(newLines) {
			diff.WriteString(fmt.Sprintf("- %s\n", oldLines[i]))
			i++
		} else if oldLines[i] == newLines[j] {
			diff.WriteString(fmt.Sprintf("  %s\n", oldLines[i]))
			i++
			j++
		} else {
			diff.WriteString(fmt.Sprintf("- %s\n", oldLines[i]))
			diff.WriteString(fmt.Sprintf("+ %s\n", newLines[j]))
			i++
			j++
		}
	}

	return diff.String()
}
