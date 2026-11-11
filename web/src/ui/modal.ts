import { esc } from "./dom";

export interface ModalAction {
  label: string;
  variant?: "acid" | "ghost" | "danger";
  onClick?: (close: () => void) => void | false;
}

export interface ModalOptions {
  kicker?: string;
  title: string;
  body: string;
  actions?: ModalAction[];
  onClose?: () => void;
  wide?: boolean;
}

let layer: HTMLElement | null = null;
let lastFocus: Element | null = null;
let escapeHandler: ((event: KeyboardEvent) => void) | null = null;

export function closeModal(): void {
  if (escapeHandler) {
    window.removeEventListener("keydown", escapeHandler);
    escapeHandler = null;
  }
  if (!layer) return;
  layer.dataset.open = "false";
  const node = layer;
  layer = null;
  document.body.classList.remove("no-scroll");
  window.setTimeout(() => node.remove(), 220);
  if (lastFocus instanceof HTMLElement) lastFocus.focus();
}

export function openModal(options: ModalOptions): HTMLElement {
  closeModal();
  lastFocus = document.activeElement;

  const root = document.createElement("div");
  root.className = "modal-layer";
  root.dataset.open = "true";
  root.innerHTML = `
    <div class="modal-backdrop" data-close></div>
    <div class="modal ${options.wide ? "modal-wide" : ""}" role="dialog" aria-modal="true" aria-label="${esc(options.title)}">
      <button class="iconbtn modal-close" data-close type="button" aria-label="Close">×</button>
      ${options.kicker ? `<span class="eyebrow">${esc(options.kicker)}</span>` : ""}
      <h2>${options.title}</h2>
      <div class="modal-body">${options.body}</div>
      ${
        options.actions && options.actions.length
          ? `<div class="modal-actions">${options.actions
              .map(
                (action, index) =>
                  `<button class="btn ${action.variant === "ghost" ? "btn-ghost" : action.variant === "danger" ? "btn-danger" : "btn-acid"}" data-action="${index}" type="button">${esc(action.label)}</button>`,
              )
              .join("")}</div>`
          : ""
      }
    </div>
  `;

  const close = () => {
    options.onClose?.();
    closeModal();
  };

  root.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    if (target.closest("[data-close]")) {
      close();
      return;
    }
    const actionIndex = target.closest<HTMLElement>("[data-action]")?.dataset.action;
    if (actionIndex === undefined) return;
    const action = options.actions?.[Number(actionIndex)];
    if (!action) return;
    const result = action.onClick?.(close);
    if (result !== false) close();
  });

  const onKey = (event: KeyboardEvent) => {
    if (event.key === "Escape") close();
  };
  escapeHandler = onKey;
  window.addEventListener("keydown", onKey);

  document.body.appendChild(root);
  document.body.classList.add("no-scroll");
  layer = root;
  window.setTimeout(() => {
    (root.querySelector("[data-action], .modal-close") as HTMLElement | null)?.focus();
  }, 40);
  return root;
}
