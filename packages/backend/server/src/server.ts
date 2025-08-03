import { NestFactory } from '@nestjs/core';
import type { NestExpressApplication } from '@nestjs/platform-express';
import cookieParser from 'cookie-parser';
import graphqlUploadExpress from 'graphql-upload/graphqlUploadExpress.mjs';

import {
  AFFiNELogger,
  CacheInterceptor,
  CloudThrottlerGuard,
  Config,
  GlobalExceptionFilter,
  URLHelper,
} from './base';
import { SocketIoAdapter } from './base/websocket';
import { AuthGuard } from './core/auth';
import { serverTimingAndCache } from './middleware/timing';

const OneMB = 1024 * 1024;

export async function run() {
  const { AppModule } = await import('./app.module');

  const app = await NestFactory.create<NestExpressApplication>(AppModule, {
    cors: true,
    rawBody: true,
    bodyParser: true,
    bufferLogs: true,
  });

  app.useBodyParser('raw', { limit: 100 * OneMB });

  const logger = app.get(AFFiNELogger);
  app.useLogger(logger);
  const config = app.get(Config);

  if (config.server.path) {
    app.setGlobalPrefix(config.server.path);
  }

  app.use(serverTimingAndCache);

  app.use(
    graphqlUploadExpress({
      maxFileSize: 100 * OneMB,
      maxFiles: 32,
    })
  );

  app.useGlobalGuards(app.get(AuthGuard), app.get(CloudThrottlerGuard));
  app.useGlobalInterceptors(app.get(CacheInterceptor));
  app.useGlobalFilters(new GlobalExceptionFilter(app.getHttpAdapter()));
  app.use(cookieParser());
  // only enable shutdown hooks in production
  // https://docs.nestjs.com/fundamentals/lifecycle-events#application-shutdown
  if (env.prod) {
    app.enableShutdownHooks();
  }

  const adapter = new SocketIoAdapter(app);
  app.useWebSocketAdapter(adapter);

  const url = app.get(URLHelper);
  const listeningHost = '0.0.0.0';

  await app.listen(config.server.port, listeningHost);

  logger.log(`AFFiNE Server is running in [${env.DEPLOYMENT_TYPE}] mode`);
  logger.log(`Listening on http://${listeningHost}:${config.server.port}`);
  logger.log(`And the public server should be recognized as ${url.baseUrl}`);
}
