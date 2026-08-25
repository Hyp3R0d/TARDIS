package com.tardis.webserver.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

public record ServerConfig(
        String host,
        int port,
        int backlog,
        int threads,
        Path webRoot,
        boolean cacheControl,
        double rateLimitRps,
        double rateLimitBurst,
        String corsOrigin,
        String adminUsername,
        String adminPassword) {

    public static ServerConfig load() throws IOException {
        Properties props = new Properties();
        try (InputStream in = ServerConfig.class.getResourceAsStream("/application.properties")) {
            if (in != null) {
                props.load(in);
            }
        }

        String host = props.getProperty("server.host", "0.0.0.0");
        int port = Integer.parseInt(props.getProperty("server.port", "8080"));
        int backlog = Integer.parseInt(props.getProperty("server.backlog", "64"));
        int threads = Integer.parseInt(props.getProperty("server.threads", "8"));
        boolean cacheControl = Boolean.parseBoolean(props.getProperty("server.cache-control", "true"));
        double rateLimitRps = Double.parseDouble(props.getProperty("server.rate-limit-rps", "20"));
        double rateLimitBurst = Double.parseDouble(props.getProperty("server.rate-limit-burst", "40"));
        String corsOrigin = props.getProperty("server.cors-origin", "*");
        String adminUsername = props.getProperty("admin.username", "admin");
        String adminPassword = props.getProperty("admin.password", "tardis2024");

        Path webRoot = Paths.get(props.getProperty("server.web-root", "static"));
        if (!Files.isDirectory(webRoot)) {
            Files.createDirectories(webRoot);
        }

        return new ServerConfig(host, port, backlog, threads,
                webRoot.toAbsolutePath().normalize(),
                cacheControl, rateLimitRps, rateLimitBurst,
                corsOrigin, adminUsername, adminPassword);
    }
}
