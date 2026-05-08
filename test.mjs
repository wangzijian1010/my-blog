import { readFile, readdir, stat } from "node:fs/promises";
import { strict as assert } from "node:assert";

async function fileExists(path) {
  try {
    const info = await stat(path);
    return info.isFile();
  } catch {
    return false;
  }
}

const index = await readFile("index.html", "utf8");
const archive = await readFile("archive.html", "utf8");
const post = await readFile("posts/interview-loop.html", "utf8");
const css = await readFile("styles.css", "utf8");
const postFiles = (await readdir("posts")).filter((file) => file.endsWith(".html"));

assert.match(index, /<link rel="stylesheet" href="styles\.css">/);
assert.match(index, /<a href="index\.html"[^>]*>Home<\/a>/);
assert.match(index, /<a href="archive\.html"[^>]*>Posts<\/a>/);
assert.match(index, /<a href="#about"[^>]*>About<\/a>/);
assert.match(index, /posts\/interview-loop\.html/);
assert.equal(await fileExists("posts/interview-loop.html"), true);
for (const file of postFiles) {
  assert.match(index, new RegExp(`posts/${file}`));
  assert.match(archive, new RegExp(`posts/${file}`));
}
assert.equal(/index\.html#posts/.test(index), false, "Posts nav should point to the all-posts page");

assert.match(post, /<article class="post">/);
assert.match(post, /面试不是考试/);
assert.match(post, /<a href="\.\.\/index\.html"/);

assert.match(css, /max-width:\s*1060px/);
assert.match(css, /max-width:\s*680px/);
assert.match(css, /prefers-color-scheme:\s*dark/);
assert.match(css, /@media\s*\(max-width:\s*760px\)/);

assert.equal(/href="#"/.test(index), false, "index.html should not contain href=\"#\" placeholders");
assert.match(index, /<!-- POSTS:START -->[\s\S]*<!-- POSTS:END -->/);

const feed = await readFile("feed.xml", "utf8");
assert.match(feed, /<atom:link[^>]+rel="self"/);
assert.match(feed, /<guid>/);

console.log("static blog checks passed");
