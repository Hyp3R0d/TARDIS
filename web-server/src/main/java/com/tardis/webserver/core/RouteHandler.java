package com.tardis.webserver.core;

import java.io.IOException;

@FunctionalInterface
public interface RouteHandler {

    void handle(RequestContext ctx) throws IOException;
}
