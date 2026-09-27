/** The workbench and static Markdown guide share stable section anchors. */
export function guideSectionId(title: string) {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
