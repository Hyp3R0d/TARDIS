package com.tardis.webserver.core;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

public final class Router implements HttpHandler {

    private final Map<String, RouteHandler> routes = new TreeMap<>();
    private RouteHandler fallback;

    public Router register(String method, String path, RouteHandler handler) {
        routes.put(method + " " + path, handler);
        return this;
    }

    public Router fallback(RouteHandler handler) {
        this.fallback = handler;
        return this;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        RequestContext ctx = new RequestContext(exchange);
        RouteHandler handler = routes.get(exchange.getRequestMethod() + " " + ctx.path());
        if (handler != null) {
            handler.handle(ctx);
            return;
        }
        if (fallback != null) {
            fallback.handle(ctx);
        } else {
            ctx.text(404, "Not Found");
        }
    }
}
