package com.tardis.webserver.util;

public final class Bytes {

    private static final String[] UNITS = {"B", "KB", "MB", "GB", "TB"};

    private Bytes() {
    }

    public static String humanReadable(long bytes) {
        if (bytes < 0) {
            return "-" + humanReadable(-bytes);
        }
        double value = bytes;
        int unit = 0;
        while (value >= 1024 && unit < UNITS.length - 1) {
            value /= 1024;
            unit++;
        }
        if (unit == 0) {
            return bytes + " B";
        }
        return String.format("%.1f %s", value, UNITS[unit]);
    }
}
