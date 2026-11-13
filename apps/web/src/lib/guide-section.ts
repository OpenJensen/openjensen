/** The workbench and static Markdown guide share stable section anchors. */
export function guideSectionId(title: string) {
  // Page names can change; a link someone saved, and the links the module list returns,
  // keep pointing at the same section.
  const stable: Record<string, string> = {
    Home: 'dashboard',
    Jobs: 'jobs',
    'AI teleoperation': 'teaching',
    'Robot Lab': 'robot-lab',
    Train: 'training',
    Simulate: 'simulation',
  };
  return (
    stable[title] ??
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  );
}
