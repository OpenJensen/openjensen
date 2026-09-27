import { publicPath } from './base-path';

export type DatasetStarter = {
  id: string;
  title: string;
  repoId: string;
  revision: string;
  description: string;
  poster: string;
  episodes: number;
  cameras: number;
};

// Counts and stills belong to these exact public dataset revisions. See
// public/datasets/README.md for sources and the preview verification record.
export const datasetStarters: DatasetStarter[] = [
  {
    id: 'so101-pickup',
    title: 'SO-101 pickup',
    repoId: 'codywang/so101_pickup_test',
    revision: 'ecef85bc07005f771ad86deeff1427f9d72953ed',
    description: 'A single arm, a cup, and a front camera.',
    poster: publicPath('/datasets/so101-pickup.jpg'),
    episodes: 30,
    cameras: 1,
  },
  {
    id: 'so100-pickplace',
    title: 'SO-100 pick & place',
    repoId: 'lerobot/svla_so100_pickplace',
    revision: '728583b5eaf9e739a7f119e2def466fa1d552402',
    description: 'Tabletop manipulation from top and wrist views.',
    poster: publicPath('/datasets/so100-pickplace.jpg'),
    episodes: 50,
    cameras: 2,
  },
];
