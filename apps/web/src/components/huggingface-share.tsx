'use client';

import { useEffect, useState, type FormEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  huggingFaceShare,
  repositoryName,
  repositoryUrl,
  resultFlag,
  resultText,
  type ShareKind,
} from '@/lib/huggingface-share';
import { isWorkActive, workJobs, type WorkJob } from '@/lib/work-jobs';
import './huggingface-share.css';

const noun = { dataset: 'dataset', model: 'model' } as const;
const page = { dataset: 'Datasets', model: 'Models' } as const;

/** Follow one publish or fetch until it ends. */
function useShareJob(projectId: string, jobId: string) {
  return useQuery({
    queryKey: ['work-job', projectId, jobId],
    queryFn: ({ signal }) => workJobs.get(projectId, jobId, signal),
    enabled: !!projectId && !!jobId,
    retry: false,
    refetchInterval: (query) => (query.state.data && !isWorkActive(query.state.data) ? false : 1000),
  });
}

function Progress({ job, projectId, starting }: { job?: WorkJob; projectId: string; starting: string }) {
  const client = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => workJobs.cancel(projectId, job!.id),
    onSuccess: (updated) => client.setQueryData(['work-job', projectId, updated.id], updated),
  });
  return (
    <div className="hf-share-progress">
      <p role="status">
        <span className="spinner" aria-hidden="true" /> {job?.message ?? starting}
      </p>
      {job?.can_cancel && (
        <button type="button" className="text-link" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
          {cancel.isPending ? 'Cancelling…' : 'Cancel'}
        </button>
      )}
    </div>
  );
}

function CopyLink({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="hf-share-link">
      <input
        readOnly
        aria-label="Link to copy"
        value={url}
        onFocus={(event) => event.currentTarget.select()}
        spellCheck={false}
      />
      <button
        type="button"
        className="secondary-button"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(url);
            setCopied(true);
          } catch {
            setCopied(false);
          }
        }}
      >
        {copied ? 'Copied' : 'Copy link'}
      </button>
    </div>
  );
}

/**
 * One button on a saved dataset or a model. It shows where the files will go, and that
 * the repository will be private, before anything is sent; afterwards it shows the link.
 */
export function PublishToHuggingFace({
  projectId,
  kind,
  id,
  blocked,
}: {
  projectId: string;
  kind: ShareKind;
  id: string;
  /** Why publishing is unavailable right now, such as unsaved changes. */
  blocked?: string;
}) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [repository, setRepository] = useState<string | null>(null);
  const [isPublic, setIsPublic] = useState(false);
  const [jobId, setJobId] = useState('');
  const [again, setAgain] = useState(false);
  const destination = useQuery({
    queryKey: ['huggingface-destination', projectId, kind, id],
    queryFn: ({ signal }) => huggingFaceShare.destination(projectId, kind, id, signal),
    enabled: open && !!projectId,
    retry: false,
    staleTime: 0,
    gcTime: 0,
  });
  // A publish started before this page was opened is picked up where it is.
  const followed = jobId || destination.data?.active_job || '';
  const job = useShareJob(projectId, followed);
  const publish = useMutation({
    mutationFn: (body: { repo_id: string; private: boolean }) =>
      huggingFaceShare.publish(projectId, { kind, id, ...body }),
    onSuccess: (started) => {
      client.setQueryData(['work-job', projectId, started.id], started);
      setJobId(started.id);
      setAgain(false);
    },
  });
  const finished = job.data && !isWorkActive(job.data) ? job.data : undefined;
  useEffect(() => {
    if (finished) void client.invalidateQueries({ queryKey: ['work-jobs', projectId] });
  }, [finished, client, projectId]);

  if (!open)
    return (
      <div className="hf-share hf-share-closed">
        <button
          type="button"
          className="secondary-button"
          disabled={!projectId || !!blocked}
          title={blocked}
          onClick={() => setOpen(true)}
        >
          Publish to Hugging Face
        </button>
      </div>
    );

  const account = destination.data?.account;
  const name = repository ?? destination.data?.repo_id ?? '';
  const running = (!!followed && (!job.data || isWorkActive(job.data))) || publish.isPending;
  const succeeded = !again && finished?.status === 'succeeded';
  const earlier = !again && !followed ? destination.data?.published : undefined;
  const link = succeeded ? resultText(finished, 'url') : earlier?.url;
  const wasPrivate = succeeded ? resultFlag(finished, 'private') : earlier?.private;
  const failure = !again && finished && finished.status !== 'succeeded' ? finished : undefined;
  const form = !running && !link;
  return (
    <section className="hf-share panel" aria-label="Publish to Hugging Face">
      <div className="hf-share-heading">
        <h3>Publish to Hugging Face</h3>
        {!running && (
          <button type="button" className="text-link" onClick={() => setOpen(false)}>
            Close
          </button>
        )}
      </div>
      {destination.isPending && <p role="status">Checking your Hugging Face account…</p>}
      {destination.isError && (
        <p className="error-notice" role="alert">
          {destination.error.message}
        </p>
      )}
      {account && !account.connected && (
        <p className="warning-box" role="alert">
          {account.message ?? 'Connect a Hugging Face account first.'} A token is saved under Cloud &amp; APIs → Hugging
          Face.
        </p>
      )}
      {running && <Progress job={job.data} projectId={projectId} starting="Starting…" />}
      {link && (
        <div className="hf-share-done">
          <p role="status">
            <strong>{wasPrivate ? 'Published privately.' : 'Published publicly.'}</strong>{' '}
            {wasPrivate
              ? 'Only your account, and people you invite on Hugging Face, can open it.'
              : 'Anyone can find and download it.'}
          </p>
          <CopyLink url={link} />
          <p className="hf-share-hint">
            On the other machine, open {page[kind]}, choose <strong>Add from a Hugging Face link</strong> and paste this
            link. {wasPrivate ? 'That machine needs the same Hugging Face account.' : ''}
          </p>
          <button
            type="button"
            className="text-link"
            onClick={() => {
              setAgain(true);
              setJobId('');
              publish.reset();
            }}
          >
            Publish again
          </button>
        </div>
      )}
      {failure && (
        <p className="error-notice" role="alert">
          {failure.message ?? 'Publishing did not finish.'}
        </p>
      )}
      {form && account?.connected && (
        <form
          className="hf-share-form"
          onSubmit={(event: FormEvent) => {
            event.preventDefault();
            if (repositoryName(name)) publish.mutate({ repo_id: name.trim(), private: !isPublic });
          }}
        >
          <p>
            Signed in as <strong>{account.username}</strong>
            {account.source === 'machine' ? ' (this computer’s hf login).' : ' (token saved in Cloud & APIs).'}
          </p>
          <label>
            Destination repository
            <input
              aria-label="Destination repository"
              value={name}
              maxLength={200}
              spellCheck={false}
              autoCapitalize="none"
              autoCorrect="off"
              onChange={(event) => setRepository(event.target.value)}
            />
          </label>
          <p className="hf-share-destination">
            {repositoryName(name) ? repositoryUrl(kind, name.trim()) : 'Use a name such as you/pick-pen.'}
          </p>
          <p className="hf-share-privacy" role="status">
            {isPublic ? (
              <>
                <strong>Public.</strong> Anyone will be able to find and download every file of this {noun[kind]}, as it
                is saved here.
              </>
            ) : (
              <>
                <strong>Private.</strong> Only your account, and people you invite on Hugging Face, will be able to open
                it.
              </>
            )}
          </p>
          <label className="hf-share-public">
            <input type="checkbox" checked={isPublic} onChange={(event) => setIsPublic(event.target.checked)} /> Make
            this repository public
          </label>
          {publish.isError && (
            <p className="error-notice" role="alert">
              {publish.error.message}
            </p>
          )}
          <button type="submit" className="primary-button" disabled={!repositoryName(name)}>
            {isPublic ? 'Publish publicly' : 'Publish privately'}
          </button>
        </form>
      )}
    </section>
  );
}

/** Paste a link, or `owner/name`, to add the dataset or model another machine published. */
export function AddFromHuggingFace({
  projectId,
  kind,
  onOpen,
}: {
  projectId: string;
  kind: ShareKind;
  /** Open what was added: the saved dataset's or the model's identity in this project. */
  onOpen: (id: string) => void;
}) {
  const client = useQueryClient();
  const [link, setLink] = useState('');
  const [jobId, setJobId] = useState('');
  const job = useShareJob(projectId, jobId);
  const add = useMutation({
    mutationFn: () => huggingFaceShare.fetch(projectId, kind, link.trim()),
    onSuccess: (started) => {
      client.setQueryData(['work-job', projectId, started.id], started);
      setJobId(started.id);
    },
  });
  const finished = job.data && !isWorkActive(job.data) ? job.data : undefined;
  const running = (!!jobId && !finished) || add.isPending;
  const added = finished?.status === 'succeeded' ? finished : undefined;
  const addedId = resultText(added, kind === 'dataset' ? 'dataset_id' : 'artifact_id');
  useEffect(() => {
    if (!finished) return;
    void client.invalidateQueries({ queryKey: [kind === 'dataset' ? 'datasets' : 'artifacts', projectId] });
    void client.invalidateQueries({ queryKey: ['work-jobs', projectId] });
  }, [finished, client, kind, projectId]);
  return (
    <form
      className="hf-add"
      aria-label="Add from a Hugging Face link"
      onSubmit={(event: FormEvent) => {
        event.preventDefault();
        if (link.trim() && !running) add.mutate();
      }}
    >
      <label htmlFor={`hf-add-${kind}`}>Add from a Hugging Face link</label>
      <div className="hf-add-row">
        <input
          id={`hf-add-${kind}`}
          value={link}
          disabled={!projectId || running}
          maxLength={500}
          placeholder={`${repositoryUrl(kind, 'you/pick-pen')} or you/pick-pen`}
          spellCheck={false}
          autoCapitalize="none"
          autoCorrect="off"
          onChange={(event) => {
            setLink(event.target.value);
            if (finished) setJobId('');
            add.reset();
          }}
        />
        <button type="submit" className="secondary-button" disabled={!projectId || !link.trim() || running}>
          {running ? 'Adding…' : `Add ${noun[kind]}`}
        </button>
      </div>
      {running && <Progress job={job.data} projectId={projectId} starting="Starting…" />}
      {add.isError && (
        <p className="error-notice" role="alert">
          {add.error.message}
        </p>
      )}
      {finished && !added && (
        <p className="error-notice" role="alert">
          {finished.message ?? `This ${noun[kind]} could not be added.`}
        </p>
      )}
      {added && (
        <p className="hf-add-done" role="status">
          <span>
            Added <strong>{resultText(added, 'name') ?? added.title}</strong>.{' '}
            {resultFlag(added, 'already_saved') ? `This project already had this ${noun[kind]}.` : ''}
          </span>
          {addedId && (
            <button type="button" className="text-link" onClick={() => onOpen(addedId)}>
              Open {noun[kind]}
            </button>
          )}
        </p>
      )}
    </form>
  );
}
