import { readFile, writeFile, readdir } from "node:fs/promises";
import { join } from "node:path";

const SITE_URL = process.env.SITE_URL ?? "https://wangzijian1010.github.io/my-blog";
const SITE_TITLE = "慢慢写";
const SITE_DESC = "面试复盘、技术观察和日常小感悟。";
const READ_CPM = 350; // chars per minute, mixed zh/en

function pick(re, html, group = 1) {
  const m = html.match(re);
  return m ? m[group].trim() : null;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function rfc822(dateStr) {
  const [y, m, d] = dateStr.split("-").map(Number);
  const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const pad = (n) => String(n).padStart(2, "0");
  const weekday = days[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
  return `${weekday}, ${pad(d)} ${months[m - 1]} ${y} 00:00:00 +0800`;
}

function formatDateDisplay(dateStr) {
  return dateStr.replaceAll("-", ".");
}

function readingMinutes(html) {
  // Extract article body, strip tags, count visible chars.
  const body = pick(/<article class="post">([\s\S]*?)<\/article>/, html) ?? html;
  const text = body.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  return Math.max(1, Math.round(text.length / READ_CPM));
}

async function loadPost(file) {
  const path = join("posts", file);
  const html = await readFile(path, "utf8");

  const titleRaw = pick(/<title>([^<]+)<\/title>/, html);
  const title = titleRaw?.split("|")[0].trim() ?? file;
  const description = pick(/<meta\s+name="description"[^>]*content="([^"]+)"/, html) ?? "";
  const date = pick(/<time\s+datetime="(\d{4}-\d{2}-\d{2})"/, html);
  if (!date) throw new Error(`${path}: missing <time datetime="YYYY-MM-DD">`);
  const metaBlock = pick(/<p class="post-meta">([\s\S]*?)<\/p>/, html) ?? "";
  const tags = [...metaBlock.matchAll(/<span>([^<]+)<\/span>/g)].map((m) => m[1].trim());
  const minutes = readingMinutes(html);

  return { file, path, title, description, date, tags, minutes };
}

function renderListItem(p) {
  const tags = p.tags.map((t) => `                <span>${escapeHtml(t)}</span>`).join("\n");
  return `          <article class="list-item">
            <time datetime="${p.date}">${formatDateDisplay(p.date)} · 约 ${p.minutes} 分钟</time>
            <div>
              <h3>
                <a href="posts/${p.file}">${escapeHtml(p.title)}</a>
              </h3>
              <p>
                ${escapeHtml(p.description)}
              </p>
              <p class="tags">
${tags}
              </p>
            </div>
          </article>`;
}

function renderFeed(posts) {
  const items = posts
    .map(
      (p) => `    <item>
      <title>${escapeHtml(p.title)}</title>
      <link>${SITE_URL}/posts/${p.file}</link>
      <guid>${SITE_URL}/posts/${p.file}</guid>
      <description>${escapeHtml(p.description)}</description>
      <pubDate>${rfc822(p.date)}</pubDate>
    </item>`,
    )
    .join("\n");
  const lastBuild = posts[0] ? rfc822(posts[0].date) : rfc822(new Date().toISOString().slice(0, 10));
  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escapeHtml(SITE_TITLE)}</title>
    <description>${escapeHtml(SITE_DESC)}</description>
    <link>${SITE_URL}/</link>
    <atom:link href="${SITE_URL}/feed.xml" rel="self" type="application/rss+xml" />
    <language>zh-CN</language>
    <lastBuildDate>${lastBuild}</lastBuildDate>
${items}
  </channel>
</rss>
`;
}

function renderSitemap(posts) {
  const urls = [
    { loc: `${SITE_URL}/`, lastmod: posts[0]?.date },
    { loc: `${SITE_URL}/archive.html`, lastmod: posts[0]?.date },
    ...posts.map((p) => ({ loc: `${SITE_URL}/posts/${p.file}`, lastmod: p.date })),
  ];
  const body = urls
    .map((u) => `  <url><loc>${u.loc}</loc>${u.lastmod ? `<lastmod>${u.lastmod}</lastmod>` : ""}</url>`)
    .join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${body}
</urlset>
`;
}

function renderRobots() {
  return `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml
`;
}

function renderArchive(posts) {
  const byYear = new Map();
  for (const p of posts) {
    const y = p.date.slice(0, 4);
    if (!byYear.has(y)) byYear.set(y, []);
    byYear.get(y).push(p);
  }
  const years = [...byYear.keys()].sort((a, b) => b.localeCompare(a));
  const sections = years
    .map((y) => {
      const items = byYear
        .get(y)
        .map(
          (p) => `        <li>
          <time datetime="${p.date}">${formatDateDisplay(p.date)}</time>
          <a href="posts/${p.file}">${escapeHtml(p.title)}</a>
        </li>`,
        )
        .join("\n");
      return `      <section>
        <h2>${y}</h2>
        <ul class="archive-list">
${items}
        </ul>
      </section>`;
    })
    .join("\n\n");

  return `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>归档 | ${SITE_TITLE}</title>
    <meta name="description" content="${SITE_TITLE} 全部文章的时间线归档。">
    <meta property="og:type" content="website">
    <meta property="og:title" content="归档 | ${SITE_TITLE}">
    <meta property="og:url" content="${SITE_URL}/archive.html">
    <meta name="twitter:card" content="summary">
    <link rel="alternate" type="application/rss+xml" title="${SITE_TITLE} RSS" href="feed.xml">
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <header class="site-header">
      <a class="brand" href="index.html">${SITE_TITLE}</a>
      <nav aria-label="Primary navigation">
        <a href="index.html">Home</a>
        <a href="archive.html" aria-current="page">Archive</a>
        <a href="index.html#about">About</a>
        <a href="feed.xml">RSS</a>
      </nav>
    </header>

    <main class="shell">
      <section class="intro">
        <p class="eyebrow">archive</p>
        <h1>全部文章</h1>
        <p>按时间倒序，记录这里写过的每一篇。</p>
      </section>

${sections}
    </main>

    <footer class="site-footer">
      <p>A quiet blog.</p>
    </footer>
  </body>
</html>
`;
}

const files = (await readdir("posts")).filter((f) => f.endsWith(".html"));
const posts = (await Promise.all(files.map(loadPost))).sort((a, b) => b.date.localeCompare(a.date));

const index = await readFile("index.html", "utf8");
const list = posts.map(renderListItem).join("\n\n");
const next = index.replace(
  /<!-- POSTS:START -->[\s\S]*?<!-- POSTS:END -->/,
  `<!-- POSTS:START -->\n${list}\n          <!-- POSTS:END -->`,
);
if (!index.includes("<!-- POSTS:START -->")) {
  throw new Error("index.html missing <!-- POSTS:START --> ... <!-- POSTS:END --> markers");
}

await writeFile("index.html", next);
await writeFile("feed.xml", renderFeed(posts));
await writeFile("sitemap.xml", renderSitemap(posts));
await writeFile("robots.txt", renderRobots());
await writeFile("archive.html", renderArchive(posts));

console.log(`built ${posts.length} post(s) → index.html, feed.xml, sitemap.xml, robots.txt, archive.html`);
