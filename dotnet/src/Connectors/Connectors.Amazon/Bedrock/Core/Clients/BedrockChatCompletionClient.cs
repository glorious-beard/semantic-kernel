// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Amazon.BedrockRuntime;
using Amazon.BedrockRuntime.Model;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.FunctionCalling;
using Microsoft.SemanticKernel.Diagnostics;
using Microsoft.SemanticKernel.Planning;

namespace Microsoft.SemanticKernel.Connectors.Amazon.Core;

/// <summary>
/// Represents a client for interacting with the chat completion through Bedrock.
/// </summary>
internal sealed class BedrockChatCompletionClient
{
    private readonly string _modelId;
    private readonly string _modelProvider;
    private readonly IAmazonBedrockRuntime _bedrockRuntime;
    private readonly IBedrockChatCompletionService _ioChatService;
    private Uri? _chatGenerationEndpoint;
    private readonly ILogger _logger;

    /// <summary>
    /// Builds the client object and registers the model input-output service given the user's passed in model ID.
    /// </summary>
    /// <param name="modelId">The model ID for the client.</param>
    /// <param name="bedrockRuntime">The <see cref="IAmazonBedrockRuntime"/> instance to be used for Bedrock runtime actions.</param>
    /// <param name="loggerFactory">The <see cref="ILoggerFactory"/> to use for logging. If null, no logging will be performed.</param>
    internal BedrockChatCompletionClient(string modelId, IAmazonBedrockRuntime bedrockRuntime, ILoggerFactory? loggerFactory = null)
    {
        var serviceFactory = new BedrockServiceFactory();
        this._modelId = modelId;
        this._bedrockRuntime = bedrockRuntime;
        this._ioChatService = serviceFactory.CreateChatCompletionService(modelId);
        this._modelProvider = serviceFactory.GetModelProviderAndName(modelId).modelProvider;
        this._logger = loggerFactory?.CreateLogger(this.GetType()) ?? NullLogger.Instance;
    }

    /// <summary>
    /// Generates a chat message based on the provided chat history and execution settings.
    /// </summary>
    /// <param name="chatHistory">The chat history to use for generating the chat message.</param>
    /// <param name="executionSettings">The execution settings for the chat completion.</param>
    /// <param name="kernel">The Semantic Kernel instance.</param>
    /// <param name="cancellationToken">The cancellation token.</param>
    /// <returns>The generated chat message.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the chat history is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the chat is empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when response content is not available.</exception>
    internal async Task<IReadOnlyList<ChatMessageContent>> GenerateChatMessageAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        Verify.NotNullOrEmpty(chatHistory);

        ConverseResponse? response = null;

        var workingHistory = new ChatHistory(chatHistory);

        var chatMessages = new List<ChatMessageContent>();

        for (var requestIndex = 0; ; ++requestIndex)
        {
            ConverseRequest converseRequest = this._ioChatService.GetConverseRequest(this._modelId, workingHistory, executionSettings);
            if (this._chatGenerationEndpoint == null)
            {
                this._chatGenerationEndpoint = new Uri(this._bedrockRuntime.DetermineServiceOperationEndpoint(converseRequest).URL);
            }
            using var activity = ModelDiagnostics.StartCompletionActivity(
                this._chatGenerationEndpoint, this._modelId, this._modelProvider, workingHistory, executionSettings);
            ActivityStatusCode activityStatus;
            var (invokeFunctionsIfPresent, cfg, toolConfig) = ConfigureToolSpec(workingHistory, kernel, executionSettings);
            converseRequest.ToolConfig = toolConfig;
            try
            {
                response = await this._bedrockRuntime.ConverseAsync(converseRequest, cancellationToken).ConfigureAwait(false);
                if ((response == null) || response.Output == null || response.Output.Message == null)
                {
                    throw new InvalidOperationException("Response failed");
                }
                if (activity is not null)
                {
                    activityStatus = BedrockClientUtilities.ConvertHttpStatusCodeToActivityStatusCode(response.HttpStatusCode);
                    activity.SetStatus(activityStatus);
                }
            }
            catch (Exception ex)
            {
                this._logger.LogError(ex, "Can't converse with '{ModelId}'. Reason: {Error}", this._modelId, ex.Message);
                if (activity is not null)
                {
                    activity.SetError(ex);
                    if (response != null)
                    {
                        activityStatus = BedrockClientUtilities.ConvertHttpStatusCodeToActivityStatusCode(response.HttpStatusCode);
                        activity.SetStatus(activityStatus);
                    }
                    else
                    {
                        // If response is null, set a default status or leave it unset
                        activity.SetStatus(ActivityStatusCode.Error); // or ActivityStatusCode.Unset
                    }
                }
                throw;
            }
            finally
            {
                activity?.SetPromptTokenUsage(response?.Usage?.InputTokens ?? default);
                activity?.SetCompletionTokenUsage(response?.Usage?.OutputTokens ?? default);
            }

            var localChatMessages = this.ConvertToMessageContent(response).ToList();

            // We're done if any of the conditions for invoking tool calls aren't met
            if (!invokeFunctionsIfPresent ||
                kernel == null ||
                executionSettings == null ||
                cfg == null ||
                response.StopReason != StopReason.Tool_use ||
                localChatMessages.All(m => m.Role != AuthorRole.Tool))
            {
                chatMessages.AddRange(localChatMessages);
                break;
            }

            var originalLength = workingHistory.Count;

            foreach (var message in localChatMessages)
            {
                if (message.Role != AuthorRole.Tool)
                {
                    workingHistory.Add(message);
                    continue;
                }

                var processor = new FunctionCallsProcessor(this._logger);
                await processor.ProcessFunctionCallsAsync(
                        message,
                        executionSettings,
                        workingHistory,
                        requestIndex,
                        _ => true, // Assume every function is advertised for the model to use
                        cfg.Options,
                        kernel,
                        isStreaming: false,
                        cancellationToken)
                    .ConfigureAwait(false);
                // NOTE: Above function also adds `message` as well as results to the history on our behalf
            }

            chatMessages.AddRange(workingHistory.Skip(originalLength));
        }

        return chatMessages;
    }

    /// <summary>
    /// Converts the ConverseResponse object as outputted by the Bedrock Runtime API call to a ChatMessageContent for the Semantic Kernel.
    /// </summary>
    /// <param name="response"> ConverseResponse object outputted by Bedrock. </param>
    /// <returns>List of ChatMessageContent objects</returns>
    private ChatMessageContent[] ConvertToMessageContent(ConverseResponse response)
    {
        if (response.Output.Message == null)
        {
            return [];
        }
        var message = response.Output.Message;
        return
        [
            new ChatMessageContent
            {
                Role = message.Content.Any(c => c.ToolUse != null) ? AuthorRole.Tool : BedrockClientUtilities.MapConversationRoleToAuthorRole(message.Role.Value),
                Items = CreateChatMessageContentItemCollection(message.Content),
                InnerContent = response
            }
        ];
    }

    private static ChatMessageContentItemCollection CreateChatMessageContentItemCollection(List<ContentBlock> contentBlocks)
    {
        var itemCollection = new ChatMessageContentItemCollection();
        itemCollection.AddRange(contentBlocks
            .Where(BedrockClientUtilities.CanConvertContentBlock)
            .Select(BedrockClientUtilities.ConvertContentBlock));
        return itemCollection;
    }

    internal async IAsyncEnumerable<StreamingChatMessageContent> StreamChatMessageAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var workingHistory = new ChatHistory(chatHistory);

        for (var requestIndex = 0; ; ++requestIndex)
        {
            var converseStreamRequest = this._ioChatService.GetConverseStreamRequest(this._modelId, workingHistory, executionSettings);
            var (invokeFunctionsIfPresent, cfg, toolConfig) = ConfigureToolSpec(workingHistory, kernel, executionSettings);
            converseStreamRequest.ToolConfig = toolConfig;
            if (this._chatGenerationEndpoint == null)
            {
                this._chatGenerationEndpoint = new Uri(this._bedrockRuntime.DetermineServiceOperationEndpoint(converseStreamRequest).URL);
            }
            ConverseStreamResponse? response = null;

            // Start completion activity with semantic kernel
            using var activity = ModelDiagnostics.StartCompletionActivity(
                this._chatGenerationEndpoint, this._modelId, this._modelProvider, chatHistory, executionSettings);
            ActivityStatusCode activityStatus;
            try
            {
                // Call converse stream async with bedrock API
                response = await this._bedrockRuntime.ConverseStreamAsync(converseStreamRequest, cancellationToken).ConfigureAwait(false);
                if (activity is not null)
                {
                    activityStatus = BedrockClientUtilities.ConvertHttpStatusCodeToActivityStatusCode(response.HttpStatusCode);
                    activity.SetStatus(activityStatus);
                }
            }
            catch (Exception ex)
            {
                this._logger.LogError(ex, "Can't converse stream with '{ModelId}'. Reason: {Error}", this._modelId, ex.Message);
                if (activity is not null)
                {
                    activity.SetError(ex);
                    if (response != null)
                    {
                        activityStatus = BedrockClientUtilities.ConvertHttpStatusCodeToActivityStatusCode(response.HttpStatusCode);
                        activity.SetStatus(activityStatus);
                    }
                    else
                    {
                        // If response is null, set a default status or leave it unset
                        activity.SetStatus(ActivityStatusCode.Error); // or ActivityStatusCode.Unset
                    }
                }
                throw;
            }
            var streamedContents = new List<StreamingChatMessageContent>();
            var toolContents = new List<StreamingFunctionCallUpdateContent>();
            var isToolCall = false;
            var toolName = string.Empty;
            var toolId = string.Empty;
            await foreach (var chunk in response.Stream.ConfigureAwait(false))
            {
                // NOTE: We're assuming that 1) the first block is a start block, 2) the last block is a stop block,
                //       3) all the rest are delta blocks, and 4) all blocks are sent in order.
                var atEnd = false;
                switch (chunk)
                {
                    case ContentBlockStartEvent e when e.Start != null:
                        // This might be a tool call
                        isToolCall = invokeFunctionsIfPresent;
                        toolName = e.Start.ToolUse.Name;
                        toolId = e.Start.ToolUse.ToolUseId;
                        break;

                    case ContentBlockStartEvent e when e.Start == null:
                        // This is a normal text block
                        isToolCall = false;
                        toolName = string.Empty;
                        toolId = string.Empty;
                        break;

                    case ContentBlockDeltaEvent e when !string.IsNullOrWhiteSpace(e.Delta.Text):
                        // This is a text block
                        var blockContent = new StreamingChatMessageContent(AuthorRole.Assistant, e.Delta.Text, e);
                        streamedContents.Add(blockContent);
                        yield return blockContent;
                        break;

                    case ContentBlockDeltaEvent e when e.Delta.ToolUse != null:
                        // This is a tool call block
                        if (isToolCall)
                        {
                            // that we will automatically execute
                            toolContents.Add(new StreamingFunctionCallUpdateContent(toolId, toolName, e.Delta.ToolUse.Input));
                        }
                        var content = new StreamingChatMessageContent(AuthorRole.Tool, e.Delta.ToolUse.Input, e);
                        streamedContents.Add(content);
                        yield return content;
                        break;

                    case ContentBlockStopEvent e:
                        atEnd = true;
                        break;

                    default:
                        // Skip unknown blocks
                        break;
                }
                if (atEnd)
                {
                    break;
                }
            }

            // End streaming activity with kernel
            activityStatus = BedrockClientUtilities.ConvertHttpStatusCodeToActivityStatusCode(response.HttpStatusCode);
            activity?.SetStatus(activityStatus);

            if (!isToolCall || kernel == null || executionSettings == null || cfg == null)
            {
                // If we are not calling a tool, we can end the activity here
                activity?.EndStreaming(streamedContents);
                yield break;
            }

            // If we are calling a tool, we need to process the tool call

            // Step 1. Create a function call message
            var inputJson = string.Concat(toolContents.Select(t => t.Arguments ?? string.Empty));
            var toolUseBlock = JsonSerializer.Deserialize<ToolUseBlock>(inputJson);
            if (toolUseBlock == null)
            {
                this._logger.LogWarning("Unable to deserialize tool use block from input JSON: {InputJson}", inputJson);
                activity?.EndStreaming(streamedContents);
                yield break;
            }
            var toolContent = (FunctionCallContent)BedrockClientUtilities.ConvertContentBlock(new ContentBlock
            {
                ToolUse = toolUseBlock,
            });
            var toolMessage = new ChatMessageContent(AuthorRole.Tool, [toolContent]);

            var originalLength = workingHistory.Count;

            // Step 2. Process the function call
            var processor = new FunctionCallsProcessor(this._logger);
            await processor.ProcessFunctionCallsAsync(
                toolMessage,
                executionSettings,
                workingHistory,
                requestIndex,
                _ => true, // Assume every function is advertised for the model to use
                cfg.Options,
                kernel,
                isStreaming: true,
                cancellationToken)
                .ConfigureAwait(false);

            // Stream out tool call results
            foreach (var message in workingHistory.Skip(originalLength))
            {
                var json = JsonSerializer.Serialize(message.Items.OfType<FunctionResultContent>().Select(r => new { callId = r.CallId, result = r.Result }));
                yield return new StreamingChatMessageContent(message.Role, json, message);
            }

            activity?.EndStreaming(streamedContents, [toolContent]);
        }
    }

    private static (bool, FunctionChoiceBehaviorConfiguration?, ToolConfiguration?) ConfigureToolSpec(ChatHistory chatHistory, Kernel? kernel, PromptExecutionSettings? executionSettings)
    {
        var invokeFunctionsIfPresent = false;
        FunctionChoiceBehaviorConfiguration? cfg = null;
        ToolConfiguration? toolConfig = null;
        if (executionSettings?.FunctionChoiceBehavior != null && kernel != null)
        {
            // Only supported if kernel is present and function choice behavior is set
            // Note that the underlying model itself may or may not support function
            // calling; this is a separate concern left as an exercise for the caller.
            var ctx = new FunctionChoiceBehaviorConfigurationContext(chatHistory)
            {
                Kernel = kernel,
            };
            cfg = executionSettings.FunctionChoiceBehavior.GetConfiguration(ctx);
            invokeFunctionsIfPresent = cfg.AutoInvoke;
            var toolSpecs = cfg.Functions?.Select(f => new { Schema = f.Metadata.ToJsonSchemaFunctionView(), Function = f }).ToList() ?? [];
            toolConfig = new ToolConfiguration
            {
                Tools = [.. toolSpecs.Select(m => new Tool
                {
                    ToolSpec = new ToolSpecification
                    {
                        Name = m.Schema.Name,
                        Description = m.Schema.Description,
                        InputSchema = new ToolInputSchema
                        {
                            Json = m.Schema.ToString(),
                        },
                    }
                })],
                ToolChoice = new ToolChoice
                {
                    Auto = cfg.Choice == FunctionChoice.Auto ? new AutoToolChoice() : null,
                    Tool = cfg.Choice == FunctionChoice.Required && toolSpecs.Count > 0 ? new SpecificToolChoice
                    {
                        // The API only supports one tool that can be forced for calling, so we take the first one
                        Name = toolSpecs.First().Schema.Name,
                    } : null,
                }
            };
        }
        return (invokeFunctionsIfPresent, cfg, toolConfig);
    }
}
