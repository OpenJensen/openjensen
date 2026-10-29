/** The workbench and static Markdown guide share stable section anchors. */
export function guideSectionId(title: string) {
  const stable: Record<string, string> = {
    Jobs: 'jobs',
    'AI teleoperation': 'teaching',
    'Initial setup': 'robot-lab',
  };
  return (
    stable[title] ??
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  );
}
