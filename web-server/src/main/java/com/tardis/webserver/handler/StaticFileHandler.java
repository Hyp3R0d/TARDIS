package com.tardis.webserver.handler;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.tardis.webserver.http.HttpResponses;
import com.tardis.webserver.http.MimeTypes;
import com.tardis.webserver.http.StatusCode;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.List;
import java.util.zip.GZIPOutputStream;

public final class StaticFileHandler implements HttpHandler {

    private static final List<String> ALLOWED_METHODS = List.of("GET", "HEAD");
    private static final List<String> COMPRESSIBLE_TYPES = List.of(
            "text/", "application/javascript", "application/json", "image/svg+xml");
    private static final int MIN_GZIP_BYTES = 512;

    private final Path webRoot;
    private final boolean cacheControlEnabled;

    public StaticFileHandler(Path webRoot, boolean cacheControlEnabled) {
        this.webRoot = webRoot.toAbsolutePath().normalize();
        this.cacheControlEnabled = cacheControlEnabled;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        if (!ALLOWED_METHODS.contains(exchange.getRequestMethod())) {
            exchange.getResponseHeaders().set("Allow", "GET, HEAD");
            HttpResponses.text(exchange, StatusCode.METHOD_NOT_ALLOWED.code(), StatusCode.METHOD_NOT_ALLOWED.reason());
            return;
        }

        String decoded = URLDecoder.decode(exchange.getRequestURI().getPath(), StandardCharsets.UTF_8);
        if (decoded.equals("/")) {
            decoded = "/index.html";
        }

        Path resolved = webRoot.resolve(decoded.substring(1)).normalize();
        if (!resolved.startsWith(webRoot)) {
            HttpResponses.text(exchange, StatusCode.FORBIDDEN.code(), StatusCode.FORBIDDEN.reason());
            return;
        }

        if (Files.isDirectory(resolved)) {
            resolved = resolved.resolve("index.html");
        }
        if (!Files.isRegularFile(resolved) || !Files.isReadable(resolved)) {
            HttpResponses.text(exchange, StatusCode.NOT_FOUND.code(), StatusCode.NOT_FOUND.reason());
            return;
        }

        String etag = etag(resolved);
        if (etag.equals(exchange.getRequestHeaders().getFirst("If-None-Match"))) {
            sendNotModified(exchange, etag);
            return;
        }

        byte[] content = Files.readAllBytes(resolved);
        String contentType = MimeTypes.forFile(resolved);
        Headers headers = exchange.getResponseHeaders();
        headers.set("ETag", etag);
        headers.set("Accept-Ranges", "bytes");
        if (cacheControlEnabled) {
            headers.set("Cache-Control", "public, max-age=3600");
        }

        String rangeHeader = exchange.getRequestHeaders().getFirst("Range");
        if (rangeHeader != null && rangeHeader.startsWith("bytes=")) {
            serveRange(exchange, content, rangeHeader);
            return;
        }

        if (content.length >= MIN_GZIP_BYTES && compressible(contentType) && acceptsGzip(exchange)) {
            content = gzip(content);
            headers.set("Content-Encoding", "gzip");
            headers.set("Vary", "Accept-Encoding");
        }

        HttpResponses.bytes(exchange, StatusCode.OK.code(), contentType, content);
    }

    private static void serveRange(HttpExchange exchange, byte[] content, String rangeHeader) throws IOException {
        long length = content.length;
        String spec = rangeHeader.substring("bytes=".length());
        int dash = spec.indexOf('-');
        long start;
        long end;
        try {
            if (dash == 0) {
                long suffix = Long.parseLong(spec.substring(1));
                start = Math.max(0, length - suffix);
                end = length - 1;
            } else if (dash == spec.length() - 1) {
                start = Long.parseLong(spec.substring(0, dash));
                end = length - 1;
            } else {
                start = Long.parseLong(spec.substring(0, dash));
                end = Math.min(Long.parseLong(spec.substring(dash + 1)), length - 1);
            }
        } catch (NumberFormatException e) {
            HttpResponses.text(exchange, StatusCode.BAD_REQUEST.code(), StatusCode.BAD_REQUEST.reason());
            return;
        }
        if (start > end || start >= length) {
            exchange.getResponseHeaders().set("Content-Range", "bytes */" + length);
            HttpResponses.text(exchange, StatusCode.RANGE_NOT_SATISFIABLE.code(), StatusCode.RANGE_NOT_SATISFIABLE.reason());
            return;
        }

        exchange.getResponseHeaders().set("Content-Range", "bytes " + start + "-" + end + "/" + length);
        if ("HEAD".equals(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(206, -1);
            exchange.close();
            return;
        }
        int len = (int) (end - start + 1);
        exchange.sendResponseHeaders(206, len);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(content, (int) start, len);
        }
    }

    private static void sendNotModified(HttpExchange exchange, String etag) throws IOException {
        exchange.getResponseHeaders().set("ETag", etag);
        exchange.sendResponseHeaders(StatusCode.NOT_MODIFIED.code(), -1);
        exchange.close();
    }

    private static String etag(Path file) throws IOException {
        long size = Files.size(file);
        FileTime mtime = Files.getLastModifiedTime(file);
        return '"' + Long.toHexString(size) + "-" + Long.toHexString(mtime.toMillis()) + '"';
    }

    private static boolean compressible(String contentType) {
        for (String prefix : COMPRESSIBLE_TYPES) {
            if (contentType.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    private static boolean acceptsGzip(HttpExchange exchange) {
        String encoding = exchange.getRequestHeaders().getFirst("Accept-Encoding");
        return encoding != null && encoding.contains("gzip");
    }

    private static byte[] gzip(byte[] data) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (GZIPOutputStream out = new GZIPOutputStream(buffer)) {
            out.write(data);
        }
        return buffer.toByteArray();
    }
}
