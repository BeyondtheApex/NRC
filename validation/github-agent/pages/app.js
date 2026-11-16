"use strict";
(() => {
  const REPO = "BeyondtheApex/NRC";
  const REF = "codex/github-agent-validation";
  const STATE_REF = "codex-agent-probe-state";
  const API = "https://api.github.com";
  let token = "";
  let activeScenario = "";
  let deviceGeneration = 0;
  const report = { kind: "github-agent-browser-probe", origin: location.origin, checks: [] };
  const el = (id) => document.getElementById(id);
  const show = (id, value) => { el(id).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2); };
  const record = (check) => { report.checks.push({ at: new Date().toISOString(), ...check }); };

  async function timedFetch(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try { return await fetch(url, { ...options, signal: controller.signal, credentials: "omit", cache: "no-store", referrerPolicy: "no-referrer" }); }
    finally { clearTimeout(timer); }
  }

  async function github(path, method = "GET", data) {
    const headers = { Accept: "application/vnd.github+json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (data !== undefined) headers["Content-Type"] = "application/json";
    const response = await timedFetch(API + path, { method, headers, ...(data === undefined ? {} : { body: JSON.stringify(data) }) });
    if (!response.ok) throw new Error(`GitHub API HTTP ${response.status}`);
    if (response.status === 204) return null;
    return response.json();
  }

  async function connect() {
    try {
      await github("/user");
      const repo = await github(`/repos/${REPO}`);
      const writable = Boolean(repo.permissions?.push || repo.permissions?.maintain || repo.permissions?.admin);
      ["start", "checkpoint", "resume"].forEach((id) => { el(id).disabled = !writable; });
      show("auth-status", writable ? "已验证用户访问和仓库写入角色。具体 API 权限仍由 GitHub 在操作时检查。" : "用户认证有效；当前仓库角色不足以提交验证操作。");
      record({ name: "authenticated-api", ok: true, repositoryWriteRole: writable });
    } catch (error) {
      token = "";
      ["start", "checkpoint", "resume"].forEach((id) => { el(id).disabled = true; });
      show("auth-status", `授权检查失败：${error.message}`);
      record({ name: "authenticated-api", ok: false, error: error.message });
    }
  }

  el("connect").addEventListener("click", async () => {
    token = el("token").value.trim();
    el("token").value = "";
    if (!token) { show("auth-status", "请在本页面输入 Token；不要发送到聊天中。"); return; }
    await connect();
  });
  el("disconnect").addEventListener("click", () => {
    token = ""; deviceGeneration += 1; el("token").value = "";
    ["start", "checkpoint", "resume"].forEach((id) => { el(id).disabled = true; });
    show("auth-status", "页面内存中的授权已清除。");
  });

  el("transport").addEventListener("click", async () => {
    el("transport").disabled = true;
    show("transport-output", "正在从浏览器发送请求……");
    const tests = [
      { name: "public-rest-repository", url: `${API}/repos/${REPO}`, options: { headers: { Accept: "application/vnd.github+json" } } },
      { name: "public-rest-workflows", url: `${API}/repos/${REPO}/actions/workflows`, options: { headers: { Accept: "application/vnd.github+json" } } },
      { name: "device-endpoint-invalid-client", url: "https://github.com/login/device/code", options: { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded" }, body: "client_id=invalid-nrc-browser-probe" } },
      { name: "token-endpoint-invalid-client", url: "https://github.com/login/oauth/access_token", options: { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded" }, body: "client_id=invalid-nrc-browser-probe&device_code=invalid&grant_type=urn:ietf:params:oauth:grant-type:device_code" } },
    ];
    const checks = await Promise.all(tests.map(async (test) => {
      try {
        const response = await timedFetch(test.url, test.options);
        await response.text();
        return { name: test.name, readableByBrowser: true, status: response.status };
      } catch (error) {
        return { name: test.name, readableByBrowser: false, error: error.name, note: "浏览器未能读取响应；需结合端点可达性区分 CORS 与网络失败。" };
      }
    }));
    checks.forEach(record);
    show("transport-output", { origin: location.origin, checks, limitation: "认证端点使用无效 Client ID；本测试不等于有效 App 的完整授权验证。" });
    el("transport").disabled = false;
  });

  async function dispatch(phase) {
    if (!token) { show("command-status", "需要先在本页面授权。"); return; }
    if (phase !== "resume") activeScenario = "probe-ui-" + Date.now();
    if (!activeScenario) { show("command-status", "本页面尚无待恢复的检查点。"); return; }
    try {
      await github(`/repos/${REPO}/actions/workflows/agent-probe-controller.yml/dispatches`, "POST", {
        ref: REF, inputs: { phase, scenario_id: activeScenario, result_json: "" },
      });
      show("command-status", `GitHub 已接受请求：${activeScenario} / ${phase}。这不代表任务已经完成；请读取状态或查看 Actions。`);
      record({ name: "workflow-dispatch", ok: true, phase, scenario: activeScenario });
    } catch (error) {
      show("command-status", `请求未获确认：${error.message}。请检查 Token 权限和工作流是否可派发。`);
      record({ name: "workflow-dispatch", ok: false, phase, error: error.message });
    }
  }
  el("start").addEventListener("click", () => dispatch("start"));
  el("checkpoint").addEventListener("click", () => dispatch("checkpoint"));
  el("resume").addEventListener("click", () => dispatch("resume"));

  el("refresh").addEventListener("click", async () => {
    try {
      const file = await github(`/repos/${REPO}/contents/validation-state/ledger.json?ref=${STATE_REF}`);
      const bytes = Uint8Array.from(atob(file.content.replace(/\s/g, "")), (c) => c.charCodeAt(0));
      show("ledger", JSON.parse(new TextDecoder().decode(bytes)));
      record({ name: "durable-state-read", ok: true, blobSha: file.sha });
    } catch (error) {
      show("ledger", `状态暂不可读：${error.message}。首次任务运行前没有账本是正常的。`);
      record({ name: "durable-state-read", ok: false, error: error.message });
    }
  });

  el("device").addEventListener("click", async () => {
    const clientId = el("client-id").value.trim();
    if (!clientId) { show("device-output", "需要你自己的、已启用 Device Flow 的 App Client ID；不接受 client secret。"); return; }
    const generation = ++deviceGeneration;
    el("verification-link").hidden = true;
    show("device-output", "请求设备码……");
    try {
      const response = await timedFetch("https://github.com/login/device/code", { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded" }, body: new URLSearchParams({ client_id: clientId }) });
      const device = await response.json();
      if (!response.ok || !device.device_code || !device.user_code) throw new Error("设备码端点未返回有效授权信息");
      show("device-output", `请在 GitHub 官方页面输入：${device.user_code}\n页面按服务器间隔轮询；可随时停止。`);
      el("verification-link").hidden = false;
      const expires = Date.now() + Number(device.expires_in || 900) * 1000;
      let interval = Math.max(5, Number(device.interval || 5));
      while (generation === deviceGeneration && Date.now() < expires) {
        await new Promise((resolve) => setTimeout(resolve, interval * 1000));
        if (generation !== deviceGeneration) return;
        const polled = await timedFetch("https://github.com/login/oauth/access_token", { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded" }, body: new URLSearchParams({ client_id: clientId, device_code: device.device_code, grant_type: "urn:ietf:params:oauth:grant-type:device_code" }) });
        const grant = await polled.json();
        if (grant.access_token) {
          if (generation !== deviceGeneration) return;
          token = grant.access_token;
          show("device-output", "已获得令牌，开始验证 API 访问。令牌不会显示或导出。");
          record({ name: "device-flow", ok: true });
          await connect();
          return;
        }
        if (grant.error === "slow_down") interval += 5;
        else if (grant.error !== "authorization_pending") throw new Error("授权未完成：" + (grant.error || "unknown_error"));
      }
      show("device-output", "设备授权已停止或过期。");
    } catch (error) {
      show("device-output", `Device Flow 未完成：${error.message}。浏览器网络错误可能来自 CORS；没有使用代理绕过。`);
      record({ name: "device-flow", ok: false, error: error.name });
    }
  });
  el("stop-device").addEventListener("click", () => { deviceGeneration += 1; show("device-output", "已停止设备授权轮询。"); });

  el("export").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = "nrc-agent-browser-probe.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();
