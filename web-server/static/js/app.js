async function main() {
    const status = document.getElementById('status');
    const info = document.getElementById('info');

    try {
        const res = await fetch('/api/health');
        const health = await res.json();
        status.textContent = '服务状态：' + health.status + '（已运行 ' + health.uptimeSeconds + ' 秒）';
        status.classList.add('ok');
    } catch (e) {
        status.textContent = '无法连接 /api/health';
        status.classList.add('bad');
    }

    try {
        const res = await fetch('/api/info');
        info.textContent = JSON.stringify(await res.json(), null, 2);
    } catch (e) {
        info.textContent = '无法连接 /api/info';
    }
}

main();
