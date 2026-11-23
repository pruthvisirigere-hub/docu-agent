# Northwind API Guide

## Authentication

All API requests must include your API key in the `X-API-Key` header. You can create a key in the developer dashboard under Settings, then API Keys. Never share your key or commit it to a repository. If a key leaks, revoke it immediately and create a new one.

## Rate Limits

The API allows 100 requests per minute per API key. If you exceed the limit, the API returns HTTP 429 and a `Retry-After` header telling you how many seconds to wait. Paid plans raise the limit to 1000 requests per minute.

## Endpoints

The base URL is `https://api.northwind.example/v1`.

- `GET /v1/users` lists users. It supports `?page=` and `?limit=` query parameters.
- `GET /v1/users/{id}` returns a single user by ID.
- `POST /v1/orders` creates a new order. The request body must include `customer_id` and `items`.

## Errors

The API uses standard HTTP status codes. A 400 means your request body is invalid, a 401 means your API key is missing or wrong, a 404 means the resource does not exist, and a 500 means something went wrong on our side. Error responses always include a JSON body with `error` and `message` fields.
