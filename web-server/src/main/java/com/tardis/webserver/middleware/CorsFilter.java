package com.tardis.webserver.middleware;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.tardis.webserver.http.StatusCode;

import java.io.IOException;

public final class CorsFilter extends Filter {

    private final String allowedOrigin;
    private final HttpHandler handler;

    public CorsFilter(String allowedOrigin, HttpHandler handler) {
        this.allowedOrigin = allowedOrigin;
        this.handler = handler;
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", allowedOrigin);
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if ("OPTIONS".equals(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(StatusCode.NO_CONTENT.code(), -1);
            exchange.close();
            return;
        }
        handler.handle(exchange);
    }

    @Override
    public String description() {
        return "CORS filter";
    }
}
