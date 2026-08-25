package com.tardis.webserver.middleware;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.util.logging.Logger;

public final class AccessLogFilter extends Filter {

    private static final Logger LOG = Logger.getLogger(AccessLogFilter.class.getName());

    private final HttpHandler handler;

    public AccessLogFilter(HttpHandler handler) {
        this.handler = handler;
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        long start = System.nanoTime();
        try {
            handler.handle(exchange);
        } finally {
            long elapsedMs = (System.nanoTime() - start) / 1_000_000;
            LOG.info(exchange.getRemoteAddress().getAddress().getHostAddress()
                    + " " + exchange.getRequestMethod()
                    + " " + exchange.getRequestURI()
                    + " " + exchange.getResponseCode()
                    + " " + elapsedMs + "ms");
        }
    }

    @Override
    public String description() {
        return "Access log filter";
    }
}
