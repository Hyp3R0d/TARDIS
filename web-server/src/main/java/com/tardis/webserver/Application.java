package com.tardis.webserver;

import com.sun.net.httpserver.HttpServer;
import com.tardis.webserver.config.ServerConfig;
import com.tardis.webserver.core.Router;
import com.tardis.webserver.handler.StaticFileHandler;
import com.tardis.webserver.http.Json;
import com.tardis.webserver.metrics.RequestMetrics;
import com.tardis.webserver.middleware.AccessLogFilter;
import com.tardis.webserver.middleware.BasicAuthFilter;
import com.tardis.webserver.middleware.CorsFilter;
import com.tardis.webserver.middleware.MetricsFilter;
import com.tardis.webserver.middleware.RateLimitFilter;
import com.tardis.webserver.module.AdminModule;
import com.tardis.webserver.module.ApiModule;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.util.concurrent.Executors;
import java.util.logging.LogManager;
import java.util.logging.Logger;

public final class Application {

    private static final Logger LOG = Logger.getLogger(Application.class.getName());

    private Application() {
    }

    public static void main(String[] args) throws Exception {
        configureLogging();
        ServerConfig config = ServerConfig.load();
        RequestMetrics metrics = new RequestMetrics();

        HttpServer server = HttpServer.create(
                new InetSocketAddress(config.host(), config.port()),
                config.backlog());

        Router apiRouter = new Router()
                .fallback(ctx -> ctx.json(404, Json.of("error", "unknown endpoint", "path", ctx.path())));
        new ApiModule(metrics).register(apiRouter);
        server.createContext("/api/", new MetricsFilter(metrics, new AccessLogFilter(
                new CorsFilter(config.corsOrigin(), new RateLimitFilter(
                        config.rateLimitRps(), config.rateLimitBurst(), apiRouter)))));

        Router adminRouter = new Router()
                .fallback(ctx -> ctx.text(404, "Not Found"));
        new AdminModule(metrics).register(adminRouter);
        server.createContext("/admin/", new MetricsFilter(metrics, new AccessLogFilter(
                new BasicAuthFilter(config.adminUsername(), config.adminPassword(), adminRouter))));

        server.createContext("/", new MetricsFilter(metrics, new AccessLogFilter(
                new StaticFileHandler(config.webRoot(), config.cacheControl()))));

        server.setExecutor(Executors.newFixedThreadPool(config.threads()));
        server.start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop(1);
            LOG.info("server stopped");
        }));

        LOG.info("TARDIS WebServer listening on " + config.host() + ":" + config.port()
                + " serving " + config.webRoot().toAbsolutePath());
    }

    private static void configureLogging() {
        try (InputStream in = Application.class.getResourceAsStream("/logging.properties")) {
            if (in != null) {
                LogManager.getLogManager().readConfiguration(in);
            }
        } catch (IOException e) {
            LOG.warning("failed to load logging.properties: " + e.getMessage());
        }
    }
}
