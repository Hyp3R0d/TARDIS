package com.tardis.webserver.core;

import com.sun.net.httpserver.HttpExchange;
import com.tardis.webserver.http.HttpResponses;
import com.tardis.webserver.http.QueryParams;

import java.io.IOException;

public final class RequestContext {

    private final HttpExchange exchange;
    private final QueryParams query;

    public RequestContext(HttpExchange exchange) {
        this.exchange = exchange;
        this.query = QueryParams.parse(exchange.getRequestURI().getRawQuery());
    }

    public HttpExchange exchange() {
        return exchange;
    }

    public QueryParams query() {
        return query;
    }

    public String path() {
        return exchange.getRequestURI().getPath();
    }

    public String method() {
        return exchange.getRequestMethod();
    }

    public String remoteAddress() {
        return exchange.getRemoteAddress().getAddress().getHostAddress();
    }

    public String header(String name) {
        return exchange.getRequestHeaders().getFirst(name);
    }

    public void json(int status, String body) throws IOException {
        HttpResponses.json(exchange, status, body);
    }

    public void text(int status, String body) throws IOException {
        HttpResponses.text(exchange, status, body);
    }
}
