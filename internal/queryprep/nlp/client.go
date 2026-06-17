package nlp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client calls the Russian NLP sidecar.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a Client pointed at addr (e.g. "http://nlp-sidecar:8001").
func NewClient(addr string, timeout time.Duration) *Client {
	return &Client{
		baseURL: addr,
		httpClient: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConnsPerHost: 32,
			},
		},
	}
}

// Analyze sends the raw query to the sidecar and returns full NLP annotation.
// Pass glinerLabels to override the sidecar's config-loaded label set for this call.
func (c *Client) Analyze(ctx context.Context, text string, withEmbedding bool, glinerLabels []string) (*AnalyzeResponse, error) {
	body, err := json.Marshal(AnalyzeRequest{
		Text:              text,
		IncludeEmbeddings: withEmbedding,
		GlinerLabels:      glinerLabels,
	})
	if err != nil {
		return nil, fmt.Errorf("nlp: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/analyze", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("nlp: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("nlp: http: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("nlp: read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("nlp: sidecar returned %d: %s", resp.StatusCode, raw)
	}

	var result AnalyzeResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("nlp: unmarshal response: %w", err)
	}

	return &result, nil
}

// ReloadLabels tells the sidecar to re-read filter configs without restarting.
func (c *Client) ReloadLabels(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/reload-labels", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("nlp: reload-labels: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("nlp: reload-labels %d: %s", resp.StatusCode, b)
	}
	return nil
}

// HealthCheck returns nil if the sidecar is reachable and healthy.
func (c *Client) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("nlp sidecar unreachable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("nlp sidecar unhealthy: status %d", resp.StatusCode)
	}
	return nil
}

// EntitiesByLabel returns all entities matching the given label (case-insensitive).
func (r *AnalyzeResponse) EntitiesByLabel(label string) []Entity {
	var out []Entity
	for _, e := range r.Entities {
		if e.Label == label {
			out = append(out, e)
		}
	}
	return out
}

// HasIntent returns true if the response contains an intent signal with confidence >= min.
func (r *AnalyzeResponse) HasIntent(name string, minConfidence float64) bool {
	for _, s := range r.IntentSignals {
		if s.Name == name && s.Confidence >= minConfidence {
			return true
		}
	}
	return false
}

// NegatedLemmas returns the set of lemmas that are under any negation scope.
func (r *AnalyzeResponse) NegatedLemmas() map[string]bool {
	out := make(map[string]bool)
	for _, ns := range r.NegationScopes {
		for _, l := range ns.ScopeLemmas {
			out[l] = true
		}
	}
	return out
}
