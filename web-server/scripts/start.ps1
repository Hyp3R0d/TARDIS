param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

Write-Host "==> mvn package"
if ($SkipTests) {
    mvn -q package -DskipTests
} else {
    mvn -q package
}

Write-Host "==> java -jar target/tardis-webserver.jar"
java -jar target/tardis-webserver.jar
