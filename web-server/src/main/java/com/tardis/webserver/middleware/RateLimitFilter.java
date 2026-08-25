package com.tardis.webserver.middleware;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.tardis.webserver.http.StatusCode;
import com.tardis.webserver.util.TokenBucket;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class RateLimitFilter extends Filter {

    private final double requestsPerSecond;
    private final double burst;
    private final HttpHandler handler;
    private final Map<String, TokenBucket> buckets = new ConcurrentHashMap<>();

    public RateLimitFilter(double requestsPerSecond, double burst, HttpHandler handler) {
        this.requestsPerSecond = requestsPerSecond;
        this.burst = burst;
        this.handler = handler;
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        String key = exchange.getRemoteAddress().getAddress().getHostAddress();
        TokenBucket bucket = buckets.computeIfAbsent(key, k -> new TokenBucket(burst, requestsPerSecond));
        if (!bucket.tryAcquire()) {
            byte[] body = "{\"error\":\"too many requests\"}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
            exchange.getResponseHeaders().set("Retry-After", "1");
            exchange.sendResponseHeaders(StatusCode.TOO_MANY_REQUESTS.code(), body.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(body);
            }
            return;
        }
        handler.handle(exchange);
    }

    @Override
    public String description() {
        return "Per-IP rate limit (" + requestsPerSecond + " rps, burst " + burst + ")";
    }
}
