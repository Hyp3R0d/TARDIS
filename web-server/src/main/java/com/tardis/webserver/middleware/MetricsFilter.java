package com.tardis.webserver.middleware;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.tardis.webserver.metrics.RequestMetrics;

import java.io.IOException;

public final class MetricsFilter extends Filter {

    private final RequestMetrics metrics;
    private final HttpHandler handler;

    public MetricsFilter(RequestMetrics metrics, HttpHandler handler) {
        this.metrics = metrics;
        this.handler = handler;
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        metrics.requestStarted();
        metrics.countPath(exchange.getRequestURI().getPath());
        try {
            handler.handle(exchange);
        } finally {
            metrics.requestFinished();
        }
    }

    @Override
    public String description() {
        return "Request metrics filter";
    }
}
