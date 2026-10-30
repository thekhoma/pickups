// One hidden <input type=file> for shots (mounted by App), so ⌘O and every "Choose file" open the same picker.
let input: HTMLInputElement | null = null;

export function registerFilePicker(el: HTMLInputElement | null): void {
  input = el;
}

export function openFilePicker(): void {
  input?.click();
}
