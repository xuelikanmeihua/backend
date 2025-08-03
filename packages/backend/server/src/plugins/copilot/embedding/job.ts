import { Injectable, Logger } from '@nestjs/common';
import { ModuleRef } from '@nestjs/core';

import {
  BlobNotFound,
  CallMetric,
  CopilotContextFileNotSupported,
  EventBus,
  JobQueue,
  mapAnyError,
  OnEvent,
  OnJob,
} from '../../../base';
import { DocReader } from '../../../core/doc';
import { readAllDocIdsFromWorkspaceSnapshot } from '../../../core/utils/blocksuite';
import { Models } from '../../../models';
import { CopilotStorage } from '../storage';
import { readStream } from '../utils';
import { getEmbeddingClient } from './client';
import type { Chunk, DocFragment } from './types';
import { EMBEDDING_DIMENSIONS, EmbeddingClient } from './types';

@Injectable()
export class CopilotEmbeddingJob {
  private readonly logger = new Logger(CopilotEmbeddingJob.name);
  private readonly workspaceJobAbortController: Map<string, AbortController> =
    new Map();

  private supportEmbedding = false;
  private client: EmbeddingClient | undefined;

  constructor(
    private readonly moduleRef: ModuleRef,
    private readonly doc: DocReader,
    private readonly event: EventBus,
    private readonly models: Models,
    private readonly queue: JobQueue,
    private readonly storage: CopilotStorage
  ) {}

  @OnEvent('config.init')
  async onConfigInit() {
    await this.setup();
  }

  @OnEvent('config.changed')
  async onConfigChanged() {
    await this.setup();
  }

  private async setup() {
    this.supportEmbedding =
      await this.models.copilotContext.checkEmbeddingAvailable();
    if (this.supportEmbedding) {
      this.client = await getEmbeddingClient(this.moduleRef);
    }
  }

  // public this client to allow overriding in tests
  get embeddingClient() {
    return this.client as EmbeddingClient;
  }

  @CallMetric('ai', 'addFileEmbeddingQueue')
  async addFileEmbeddingQueue(file: Jobs['copilot.embedding.files']) {
    if (!this.supportEmbedding) return;

    const { userId, workspaceId, contextId, blobId, fileId, fileName } = file;
    await this.queue.add('copilot.embedding.files', {
      userId,
      workspaceId,
      contextId,
      blobId,
      fileId,
      fileName,
    });
  }

  @OnEvent('workspace.doc.embedding')
  async addDocEmbeddingQueue(
    docs: Events['workspace.doc.embedding'],
    options?: { contextId: string; priority: number }
  ) {
    if (!this.supportEmbedding) return;

    for (const { workspaceId, docId } of docs) {
      const jobId = `workspace:embedding:${workspaceId}:${docId}`;
      const job = await this.queue.get(jobId, 'copilot.embedding.docs');
      // if the job exists and is older than 5 minute, remove it
      if (job && job.timestamp + 5 * 60 * 1000 < Date.now()) {
        this.logger.verbose(`Removing old embedding job ${jobId}`);
        await this.queue.remove(jobId, 'copilot.embedding.docs');
      }

      await this.queue.add(
        'copilot.embedding.docs',
        {
          contextId: options?.contextId,
          workspaceId,
          docId,
        },
        {
          jobId: `workspace:embedding:${workspaceId}:${docId}`,
          priority: options?.priority ?? 1,
          timestamp: Date.now(),
        }
      );
    }
  }

  @OnEvent('workspace.updated')
  async onWorkspaceConfigUpdate({
    id,
    enableDocEmbedding,
  }: Events['workspace.updated']) {
    // trigger workspace embedding
    this.event.emit('workspace.embedding', {
      workspaceId: id,
      enableDocEmbedding,
    });
  }

  @OnEvent('workspace.embedding')
  async addWorkspaceEmbeddingQueue({
    workspaceId,
    enableDocEmbedding,
  }: Events['workspace.embedding']) {
    if (!this.supportEmbedding || !this.embeddingClient) return;

    if (enableDocEmbedding === undefined) {
      enableDocEmbedding =
        await this.models.workspace.allowEmbedding(workspaceId);
    }

    if (enableDocEmbedding) {
      const toBeEmbedDocIds =
        await this.models.copilotWorkspace.findDocsToEmbed(workspaceId);
      if (!toBeEmbedDocIds.length) {
        return;
      }
      // filter out trashed docs
      const rootSnapshot = await this.models.doc.getSnapshot(
        workspaceId,
        workspaceId
      );
      if (!rootSnapshot) {
        this.logger.warn(
          `Root snapshot for workspace ${workspaceId} not found, skipping embedding.`
        );
        return;
      }
      const allDocIds = new Set(
        readAllDocIdsFromWorkspaceSnapshot(rootSnapshot.blob)
      );
      this.logger.log(
        `Trigger embedding for ${toBeEmbedDocIds.length} docs in workspace ${workspaceId}`
      );
      const finalToBeEmbedDocIds = toBeEmbedDocIds.filter(docId =>
        allDocIds.has(docId)
      );
      for (const docId of finalToBeEmbedDocIds) {
        await this.queue.add(
          'copilot.embedding.docs',
          {
            workspaceId,
            docId,
          },
          {
            jobId: `workspace:embedding:${workspaceId}:${docId}`,
            priority: 1,
          }
        );
      }
    } else {
      const controller = this.workspaceJobAbortController.get(workspaceId);
      if (controller) {
        controller.abort();
        this.workspaceJobAbortController.delete(workspaceId);
      }
    }
  }

  @OnJob('copilot.embedding.updateDoc')
  async addDocEmbeddingQueueFromEvent(
    doc: Jobs['copilot.embedding.updateDoc']
  ) {
    if (!this.supportEmbedding || !this.embeddingClient) return;

    await this.queue.add(
      'copilot.embedding.docs',
      {
        workspaceId: doc.workspaceId,
        docId: doc.docId,
      },
      {
        jobId: `workspace:embedding:${doc.workspaceId}:${doc.docId}`,
        priority: 2,
      }
    );
  }

  @OnJob('copilot.embedding.deleteDoc')
  async deleteDocEmbeddingQueueFromEvent(
    doc: Jobs['copilot.embedding.deleteDoc']
  ) {
    await this.queue.remove(
      `workspace:embedding:${doc.workspaceId}:${doc.docId}`,
      'copilot.embedding.docs'
    );
    await this.models.copilotContext.deleteWorkspaceEmbedding(
      doc.workspaceId,
      doc.docId
    );
  }

  private async readCopilotBlob(
    userId: string,
    workspaceId: string,
    blobId: string,
    fileName: string
  ) {
    const { body } = await this.storage.get(userId, workspaceId, blobId);
    if (!body) throw new BlobNotFound({ spaceId: workspaceId, blobId });
    const buffer = await readStream(body);
    return new File([buffer], fileName);
  }

  @OnJob('copilot.embedding.files')
  async embedPendingFile({
    userId,
    workspaceId,
    contextId,
    blobId,
    fileId,
    fileName,
  }: Jobs['copilot.embedding.files']) {
    if (!this.supportEmbedding || !this.embeddingClient) return;

    try {
      const file = await this.readCopilotBlob(
        userId,
        workspaceId,
        blobId,
        fileName
      );

      // no need to check if embeddings is empty, will throw internally
      const chunks = await this.embeddingClient.getFileChunks(file);
      const total = chunks.reduce((acc, c) => acc + c.length, 0);

      for (const chunk of chunks) {
        const embeddings = await this.embeddingClient.generateEmbeddings(chunk);
        if (contextId) {
          // for context files
          await this.models.copilotContext.insertFileEmbedding(
            contextId,
            fileId,
            embeddings
          );
        } else {
          // for workspace files
          await this.models.copilotWorkspace.insertFileEmbeddings(
            workspaceId,
            fileId,
            embeddings
          );
        }
      }

      if (contextId) {
        this.event.emit('workspace.file.embed.finished', {
          contextId,
          fileId,
          chunkSize: total,
        });
      }
    } catch (error: any) {
      if (contextId) {
        this.event.emit('workspace.file.embed.failed', {
          contextId,
          fileId,
          error: mapAnyError(error).message,
        });
      }

      // passthrough error to job queue
      throw error;
    }
  }

  private async getDocFragment(
    workspaceId: string,
    docId: string
  ): Promise<DocFragment | null> {
    const docContent = await this.doc.getFullDocContent(workspaceId, docId);
    const authors = await this.models.doc.getAuthors(workspaceId, docId);
    if (docContent && authors) {
      const { title = 'Untitled', summary } = docContent;
      const { createdAt, updatedAt, createdByUser, updatedByUser } = authors;
      return {
        title,
        summary,
        createdAt: createdAt.toDateString(),
        updatedAt: updatedAt.toDateString(),
        createdBy: createdByUser?.name,
        updatedBy: updatedByUser?.name,
      };
    }
    return null;
  }

  private formatDocChunks(chunks: Chunk[], fragment: DocFragment): Chunk[] {
    return chunks.map(chunk => ({
      index: chunk.index,
      content: [
        `Title: ${fragment.title}`,
        `Created at: ${fragment.createdAt}`,
        `Updated at: ${fragment.updatedAt}`,
        fragment.createdBy ? `Created by: ${fragment.createdBy}` : undefined,
        fragment.updatedBy ? `Updated by: ${fragment.updatedBy}` : undefined,
        chunk.content,
      ]
        .filter(Boolean)
        .join('\n'),
    }));
  }

  private getWorkspaceSignal(workspaceId: string) {
    let controller = this.workspaceJobAbortController.get(workspaceId);
    if (!controller) {
      controller = new AbortController();
      this.workspaceJobAbortController.set(workspaceId, controller);
    }
    return controller.signal;
  }

  private async fulfillEmptyEmbedding(workspaceId: string, docId: string) {
    const emptyEmbedding = {
      index: 0,
      content: '',
      embedding: Array.from({ length: EMBEDDING_DIMENSIONS }, () => 0),
    };
    await this.models.copilotContext.insertWorkspaceEmbedding(
      workspaceId,
      docId,
      [emptyEmbedding]
    );
  }

  @OnJob('copilot.embedding.docs')
  async embedPendingDocs({
    contextId,
    workspaceId,
    docId,
  }: Jobs['copilot.embedding.docs']) {
    if (!this.supportEmbedding || !this.embeddingClient) return;
    if (workspaceId === docId || docId.includes('$')) return;
    const signal = this.getWorkspaceSignal(workspaceId);

    try {
      const hasNewDoc = await this.models.doc.exists(
        workspaceId,
        docId.split(':space:')[1] || ''
      );
      const needEmbedding =
        await this.models.copilotWorkspace.checkDocNeedEmbedded(
          workspaceId,
          docId
        );
      this.logger.log(
        `Check if doc ${docId} in workspace ${workspaceId} needs embedding: ${needEmbedding}`
      );
      if (needEmbedding) {
        if (signal.aborted) {
          this.logger.log(
            `Doc ${docId} in workspace ${workspaceId} is aborted, skipping embedding.`
          );
          return;
        }
        // if doc id deprecated, skip embedding and fulfill empty embedding
        const fragment = !hasNewDoc
          ? await this.getDocFragment(workspaceId, docId)
          : undefined;
        if (!hasNewDoc && fragment) {
          // fast fall for empty doc, journal is easily to create a empty doc
          if (fragment.summary.trim()) {
            const embeddings = await this.embeddingClient.getFileEmbeddings(
              new File(
                [fragment.summary],
                `${fragment.title || 'Untitled'}.md`
              ),
              chunks => this.formatDocChunks(chunks, fragment),
              signal
            );

            for (const chunks of embeddings) {
              await this.models.copilotContext.insertWorkspaceEmbedding(
                workspaceId,
                docId,
                chunks
              );
            }
            this.logger.log(
              `Doc ${docId} in workspace ${workspaceId} has summary, embedding done.`
            );
          } else {
            // for empty doc, insert empty embedding
            this.logger.warn(
              `Doc ${docId} in workspace ${workspaceId} has no summary, fulfilling empty embedding.`
            );
            await this.fulfillEmptyEmbedding(workspaceId, docId);
          }
        } else {
          this.logger.warn(
            `Doc ${docId} in workspace ${workspaceId} has no fragment, fulfilling empty embedding.`
          );
          await this.fulfillEmptyEmbedding(workspaceId, docId);
        }
      }
    } catch (error: any) {
      if (contextId) {
        this.event.emit('workspace.doc.embed.failed', {
          contextId,
          docId,
        });
      }
      if (
        error instanceof CopilotContextFileNotSupported &&
        error.message.includes('no content found')
      ) {
        this.logger.warn(
          `Doc ${docId} in workspace ${workspaceId} has no content, fulfilling empty embedding.`
        );
        // if the doc is empty, we still need to fulfill the embedding
        await this.fulfillEmptyEmbedding(workspaceId, docId);
        return;
      }

      // log error and skip the job
      this.logger.error(
        `Error embedding doc ${docId} in workspace ${workspaceId}`,
        error
      );
    }
  }

  @OnJob('copilot.embedding.cleanupTrashedDocEmbeddings')
  async cleanupTrashedDocEmbeddings({
    workspaceId,
  }: Jobs['copilot.embedding.cleanupTrashedDocEmbeddings']) {
    const workspace = await this.models.workspace.get(workspaceId);
    if (!workspace) {
      this.logger.warn(`workspace ${workspaceId} not found`);
      return;
    }

    const snapshot = await this.models.doc.getSnapshot(
      workspaceId,
      workspaceId
    );
    if (!snapshot) {
      this.logger.warn(`workspace snapshot ${workspaceId} not found`);
      return;
    }

    const docIdsInWorkspace = readAllDocIdsFromWorkspaceSnapshot(snapshot.blob);
    const docIdsInEmbedding =
      await this.models.copilotContext.listWorkspaceEmbedding(workspaceId);
    const docIdsInWorkspaceSet = new Set(docIdsInWorkspace);

    const deletedDocIds = docIdsInEmbedding.filter(
      docId => !docIdsInWorkspaceSet.has(docId)
    );
    for (const docId of deletedDocIds) {
      await this.models.copilotContext.deleteWorkspaceEmbedding(
        workspaceId,
        docId
      );
    }
  }
}
