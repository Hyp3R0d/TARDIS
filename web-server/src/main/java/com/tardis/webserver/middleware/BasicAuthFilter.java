package com.tardis.webserver.middleware;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.tardis.webserver.http.StatusCode;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

public final class BasicAuthFilter extends Filter {

    private final String expected;
    private final HttpHandler handler;

    public BasicAuthFilter(String username, String password, HttpHandler handler) {
        this.expected = "Basic " + Base64.getEncoder()
                .encodeToString((username + ":" + password).getBytes(StandardCharsets.UTF_8));
        this.handler = handler;
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        String header = exchange.getRequestHeaders().getFirst("Authorization");
        if (header == null || !header.equals(expected)) {
            byte[] body = "{\"error\":\"unauthorized\"}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
            exchange.getResponseHeaders().set("WWW-Authenticate", "Basic realm=\"admin\", charset=\"UTF-8\"");
            exchange.sendResponseHeaders(StatusCode.UNAUTHORIZED.code(), body.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(body);
            }
            return;
        }
        handler.handle(exchange);
    }

    @Override
    public String description() {
        return "HTTP Basic auth filter";
    }
}
