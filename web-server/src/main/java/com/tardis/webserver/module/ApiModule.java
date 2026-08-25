package com.tardis.webserver.module;

import com.tardis.webserver.core.RouteModule;
import com.tardis.webserver.core.Router;
import com.tardis.webserver.http.Json;
import com.tardis.webserver.metrics.RequestMetrics;

import java.net.InetAddress;

public final class ApiModule implements RouteModule {

    private final RequestMetrics metrics;

    public ApiModule(RequestMetrics metrics) {
        this.metrics = metrics;
    }

    @Override
    public void register(Router router) {
        router.register("GET", "/api/health", ctx -> ctx.json(200, Json.of(
                "status", "UP",
                "uptimeSeconds", metrics.uptimeSeconds())));

        router.register("GET", "/api/info", ctx -> ctx.json(200, Json.of(
                "name", "tardis-webserver",
                "version", "1.1.0",
                "hostname", hostname(),
                "java", System.getProperty("java.version"),
                "os", System.getProperty("os.name"),
                "arch", System.getProperty("os.arch"))));

        router.register("GET", "/api/echo", ctx -> ctx.json(200, Json.of(
                "echo", ctx.query().get("msg", ""))));

        router.register("GET", "/api/metrics", ctx -> ctx.json(200, Json.of(
                "totalRequests", metrics.totalRequests(),
                "activeRequests", metrics.activeRequests(),
                "uptimeSeconds", metrics.uptimeSeconds(),
                "paths", metrics.perPathCounts())));
    }

    private static String hostname() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            return "unknown";
        }
    }
}
