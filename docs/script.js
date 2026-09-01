const copyText = async (text, button) => {
  try {
    await navigator.clipboard.writeText(text);
    const label = button.querySelector("span");
    const original = label?.textContent;

    button.classList.add("copied");
    if (label) label.textContent = "Copied";
    button.setAttribute("aria-label", "Copied to clipboard");

    window.setTimeout(() => {
      button.classList.remove("copied");
      if (label) label.textContent = original;
      button.removeAttribute("aria-label");
    }, 1600);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    const source = document.createElement("span");

    source.textContent = text;
    source.style.position = "fixed";
    source.style.opacity = "0";
    document.body.appendChild(source);
    range.selectNodeContents(source);
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
    source.remove();
  }
};

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => {
    copyText(button.dataset.copy, button);
  });
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.querySelector(button.dataset.copyTarget);
    if (!target) return;

    copyText(target.textContent, button);
  });
});

const knotInput = document.querySelector("#knots");
const knotLabel = document.querySelector("#knot-label");
const splinePath = document.querySelector("#spline-path");

const splineShapes = [
  "M58 310 C112 297 132 257 181 231 C243 198 279 174 330 132 C385 87 430 62 492 40",
  "M58 310 C100 299 120 275 151 251 C188 222 207 224 240 204 C280 179 299 151 330 132 C372 106 409 99 438 68 C459 47 476 42 492 40",
  "M58 310 C100 298 115 280 145 257 C188 224 203 222 238 206 C280 187 298 153 330 132 C370 106 402 102 430 75 C456 51 473 43 492 40",
  "M58 310 C92 303 112 277 141 261 C175 241 201 216 234 209 C271 201 292 148 330 132 C368 116 391 111 423 80 C451 53 474 45 492 40",
  "M58 310 C87 306 108 281 139 263 C173 243 198 215 232 211 C270 207 289 145 330 132 C367 120 389 113 420 82 C448 55 474 46 492 40",
];

const updateSpline = () => {
  const value = Number(knotInput.value);
  const progress = ((value - Number(knotInput.min)) / (Number(knotInput.max) - Number(knotInput.min))) * 100;
  const shapeIndex = Math.min(
    splineShapes.length - 1,
    Math.round(((value - Number(knotInput.min)) / (Number(knotInput.max) - Number(knotInput.min))) * (splineShapes.length - 1)),
  );

  knotLabel.textContent = `${value} knots`;
  knotInput.style.background = `linear-gradient(to right, var(--accent) ${progress}%, var(--inset) ${progress}%)`;
  splinePath.setAttribute("d", splineShapes[shapeIndex]);
};

knotInput.addEventListener("input", updateSpline);
updateSpline();
