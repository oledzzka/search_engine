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

// Client calls the Python NLP sidecar.
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
func (c *Client) Analyze(ctx context.Context, text string, withEmbedding bool) (*AnalyzeResponse, error) {
	body, err := json.Marshal(AnalyzeRequest{
		Text:              text,
		IncludeEmbeddings: withEmbedding,
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

// HealthCheck returns true if the sidecar is reachable and healthy.
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
