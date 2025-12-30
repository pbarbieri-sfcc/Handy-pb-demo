use crate::settings::PostProcessProvider;
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, REFERER, USER_AGENT};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

/// Build headers for API requests based on provider type
fn build_headers(provider: &PostProcessProvider, api_key: &str) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();

    // Common headers
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));

    // Provider-specific auth headers
    if !api_key.is_empty() {
        if provider.id == "anthropic" {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| format!("Invalid API key header value: {}", e))?,
            );
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        } else {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| format!("Invalid authorization header value: {}", e))?,
            );
        }
    }

    Ok(headers)
}

/// Create an HTTP client with provider-specific headers
fn create_client(provider: &PostProcessProvider, api_key: &str) -> Result<reqwest::Client, String> {
    let headers = build_headers(provider, api_key)?;
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

/// Send a chat completion request to an OpenAI-compatible API
/// Returns Ok(Some(content)) on success, Ok(None) if response has no content,
/// or Err on actual errors (HTTP, parsing, etc.)
pub async fn send_chat_completion(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    prompt: String,
) -> Result<Option<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');

    // Validate base URL
    if base_url.is_empty() {
        return Err("Base URL is empty. Please configure the provider's base URL.".to_string());
    }

    let url = format!("{}/chat/completions", base_url);

    debug!("Sending chat completion request to: {}", url);

    let client = create_client(provider, &api_key)?;

    let request_body = ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }],
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            // Provide more specific error messages based on error type
            if e.is_timeout() {
                format!("Request timed out after 30 seconds. Please check your connection and try again.")
            } else if e.is_connect() {
                format!("Failed to connect to {}. Please check the base URL and your internet connection.", base_url)
            } else if e.is_request() {
                format!("Invalid request: {}", e)
            } else {
                format!("Network error: {}", e)
            }
        })?;

    let status = response.status();
    if !status.is_success() {
        // Provide specific error messages for common auth failures
        if status == 401 {
            return Err(
                "Authentication failed. Please check your API key and try again.".to_string(),
            );
        } else if status == 403 {
            return Err(
                "Access forbidden. Your API key may not have permission to access this resource."
                    .to_string(),
            );
        } else if status == 404 {
            return Err(format!(
                "API endpoint not found ({}). Please verify the base URL is correct.",
                base_url
            ));
        }

        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error response".to_string());
        return Err(format!(
            "API request failed with status {}: {}",
            status, error_text
        ));
    }

    let completion: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse API response: {}", e))?;

    Ok(completion
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone()))
}

/// Fetch available models from an OpenAI-compatible API
/// Returns a list of model IDs
pub async fn fetch_models(
    provider: &PostProcessProvider,
    api_key: String,
) -> Result<Vec<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');

    // Validate base URL
    if base_url.is_empty() {
        return Err("Base URL is empty. Please configure the provider's base URL.".to_string());
    }

    let url = format!("{}/models", base_url);

    debug!("Fetching models from: {}", url);

    let client = create_client(provider, &api_key)?;

    let response = client.get(&url).send().await.map_err(|e| {
        // Provide more specific error messages based on error type
        if e.is_timeout() {
            format!(
                "Request timed out after 30 seconds. Please check your connection and try again."
            )
        } else if e.is_connect() {
            format!(
                "Failed to connect to {}. Please check the base URL and your internet connection.",
                base_url
            )
        } else if e.is_request() {
            format!("Invalid request: {}", e)
        } else {
            format!("Network error: {}", e)
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        // Provide specific error messages for common auth failures
        if status == 401 {
            return Err(
                "Authentication failed. Please check your API key and try again.".to_string(),
            );
        } else if status == 403 {
            return Err(
                "Access forbidden. Your API key may not have permission to access this resource."
                    .to_string(),
            );
        } else if status == 404 {
            return Err(format!(
                "API endpoint not found ({}). Please verify the base URL is correct.",
                base_url
            ));
        }

        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!(
            "Model list request failed ({}): {}",
            status, error_text
        ));
    }

    let parsed: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let mut models = Vec::new();

    // Handle OpenAI format: { data: [ { id: "..." }, ... ] }
    if let Some(data) = parsed.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            } else if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                models.push(name.to_string());
            }
        }
    }
    // Handle array format: [ "model1", "model2", ... ]
    else if let Some(array) = parsed.as_array() {
        for entry in array {
            if let Some(model) = entry.as_str() {
                models.push(model.to_string());
            }
        }
    }

    // Warn if no models were found
    if models.is_empty() {
        return Err("No models found in the response. The API may not be compatible or may require additional configuration.".to_string());
    }

    Ok(models)
}
