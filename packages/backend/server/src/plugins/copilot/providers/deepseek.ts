import {
  createOpenAICompatible,
  OpenAICompatibleProvider as VercelOpenAICompatibleProvider,
} from '@ai-sdk/openai-compatible';
import { AISDKError, generateText, streamText } from 'ai';

import {
  CopilotProviderSideError,
  metrics,
  UserFriendlyError,
} from '../../../base';
import { CopilotProvider } from './provider';
import type {
  CopilotChatOptions,
  ModelConditions,
  PromptMessage,
  StreamObject,
} from './types';
import { CopilotProviderType, ModelInputType, ModelOutputType } from './types';
import {
  chatToGPTMessage,
  StreamObjectParser,
  TextStreamParser,
} from './utils';

export type DeepSeekConfig = {
  apiKey?: string;
};

export class DeepSeekProvider extends CopilotProvider<DeepSeekConfig> {
  readonly type = CopilotProviderType.DeepSeek; // 枚举中需要新增 DeepSeek

  // DeepSeek v3 系列模型（示例，可根据官方文档调整）
  readonly models = [
    {
      id: 'deepseek-chat',
      capabilities: [
        {
          input: [ModelInputType.Text],
          output: [ModelOutputType.Text, ModelOutputType.Object],
        },
      ],
    },
    {
      id: 'deepseek-coder-v3',
      capabilities: [
        {
          input: [ModelInputType.Text],
          output: [ModelOutputType.Text, ModelOutputType.Object],
        },
      ],
    },
  ];

  #instance!: VercelOpenAICompatibleProvider;

  override configured(): boolean {
    return !!this.config.apiKey;
  }

  protected override setup() {
    super.setup();
    this.#instance = createOpenAICompatible({
      name: this.type,
      apiKey: this.config.apiKey,
      // DeepSeek OpenAI 兼容接口
      baseURL: 'https://api.deepseek.com/v1',
    });
  }

  private handleError(e: any) {
    if (e instanceof UserFriendlyError) {
      return e;
    } else if (e instanceof AISDKError) {
      return new CopilotProviderSideError({
        provider: this.type,
        kind: e.name || 'unknown',
        message: e.message,
      });
    } else {
      return new CopilotProviderSideError({
        provider: this.type,
        kind: 'unexpected_response',
        message: e?.message || 'Unexpected deepseek response',
      });
    }
  }

  /**
   * 一次性返回完整文本
   */
  async text(
    cond: ModelConditions,
    messages: PromptMessage[],
    options: CopilotChatOptions = {}
  ): Promise<string> {
    const fullCond = {
      ...cond,
      outputType: ModelOutputType.Text,
    };
    await this.checkParams({ messages, cond: fullCond, options });
    const model = this.selectModel(fullCond);

    try {
      metrics.ai.counter('chat_text_calls').add(1, { model: model.id });

      const [system, msgs] = await chatToGPTMessage(messages);

      const modelInstance = this.#instance(model.id);

      const { text } = await generateText({
        model: modelInstance,
        system,
        messages: msgs,
        abortSignal: options.signal,
      });

      return text.trim();
    } catch (e: any) {
      metrics.ai.counter('chat_text_errors').add(1, { model: model.id });
      throw this.handleError(e);
    }
  }

  /**
   * 流式文本生成
   */
  async *streamText(
    cond: ModelConditions,
    messages: PromptMessage[],
    options: CopilotChatOptions = {}
  ): AsyncIterable<string> {
    const fullCond = {
      ...cond,
      outputType: ModelOutputType.Text,
    };
    await this.checkParams({ messages, cond: fullCond, options });
    const model = this.selectModel(fullCond);

    try {
      metrics.ai.counter('chat_text_stream_calls').add(1, { model: model.id });
      const [system, msgs] = await chatToGPTMessage(messages);

      const modelInstance = this.#instance(model.id);

      const { fullStream } = streamText({
        model: modelInstance,
        system,
        messages: msgs,
        abortSignal: options.signal,
      });

      const textParser = new TextStreamParser();
      for await (const chunk of fullStream) {
        switch (chunk.type) {
          case 'text-delta': {
            const result = textParser.parse(chunk);
            yield result;
            break;
          }
          default: {
            yield textParser.parse(chunk);
            break;
          }
        }
        if (options.signal?.aborted) {
          await fullStream.cancel();
          break;
        }
      }
    } catch (e: any) {
      metrics.ai.counter('chat_text_stream_errors').add(1, { model: model.id });
      throw this.handleError(e);
    }
  }

  /**
   * 流式对象生成（结构化 JSON）
   */
  override async *streamObject(
    cond: ModelConditions,
    messages: PromptMessage[],
    options: CopilotChatOptions = {}
  ): AsyncIterable<StreamObject> {
    const fullCond = {
      ...cond,
      outputType: ModelOutputType.Object,
    };
    await this.checkParams({ messages, cond: fullCond, options });
    const model = this.selectModel(fullCond);

    try {
      metrics.ai
        .counter('chat_object_stream_calls')
        .add(1, { model: model.id });

      const [system, msgs] = await chatToGPTMessage(messages);

      const modelInstance = this.#instance(model.id);

      const { fullStream } = streamText({
        model: modelInstance,
        system,
        messages: msgs,
        abortSignal: options.signal,
      });

      const objectParser = new StreamObjectParser();

      for await (const chunk of fullStream) {
        const obj = objectParser.parse(chunk);
        if (obj) {
          yield obj;
        }
        if (options.signal?.aborted) {
          await fullStream.cancel();
          break;
        }
      }
    } catch (e: any) {
      metrics.ai
        .counter('chat_object_stream_errors')
        .add(1, { model: model.id });
      throw this.handleError(e);
    }
  }
}
