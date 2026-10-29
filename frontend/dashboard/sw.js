// API requests are deliberately excluded; prediction and history responses must stay live.
const CACHE_NAME = "bfd-shell-v18";
const SHELL = [
  "./",
  "index.html",
  "manifest.webmanifest?v=bfd2",
  "icon.svg",
  "icon-192.svg",
  "icon-512.svg",
  "tokens.css",
  "dashboard.css"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL))
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("message", event => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;

  if (e.request.mode === "navigate") {
    e.respondWith(
      fetch(e.request)
        .then(response => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put("./", copy));
          }
          return response;
        })
        .catch(async () => (await caches.match(e.request)) || (await caches.match("index.html")))
    );
    return;
  }

  if (!SHELL.some(path => new URL(path, self.registration.scope).pathname === url.pathname)) return;

  e.respondWith(
    caches.match(e.request).then(cached =>
      cached ||
      fetch(e.request).then(response => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(e.request, copy));
        }
        return response;
      })
    )
  );
});
