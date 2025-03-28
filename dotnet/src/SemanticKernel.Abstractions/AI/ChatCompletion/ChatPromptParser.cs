﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;

namespace Microsoft.SemanticKernel.ChatCompletion;

/// <summary>
/// Chat Prompt parser.
/// </summary>
internal static class ChatPromptParser
{
    private const string MessageTagName = "message";
    private const string RoleAttributeName = "role";
    private const string ImageTagName = "image";
    private const string TextTagName = "text";
    private const string AudioTagName = "audio";
    private const string PdfTagName = "pdf";
    private const string DocxTagName = "docx";
    private const string DocTagName = "doc";
    private const string BinaryTagName = "file";

    /// <summary>
    /// Parses a prompt for an XML representation of a <see cref="ChatHistory"/>.
    /// </summary>
    /// <param name="prompt">The prompt to parse.</param>
    /// <param name="chatHistory">The parsed <see cref="ChatHistory"/>, or null if it couldn't be parsed.</param>
    /// <returns>true if the history could be parsed; otherwise, false.</returns>
    public static bool TryParse(string prompt, [NotNullWhen(true)] out ChatHistory? chatHistory)
    {
        // Parse the input string into nodes and then those nodes into a chat history.
        // The XML parsing is expensive, so we do a quick up-front check to make sure
        // the text contains "<message", as that's required in any valid XML prompt.
        const string MessageTagStart = "<" + MessageTagName;
        if (prompt is not null &&
#if NET
            prompt.Contains(MessageTagStart, StringComparison.OrdinalIgnoreCase) &&
#else
            prompt.IndexOf(MessageTagStart, StringComparison.OrdinalIgnoreCase) >= 0 &&
#endif
            XmlPromptParser.TryParse(prompt, out var nodes) &&
            TryParse(nodes, out chatHistory))
        {
            return true;
        }

        chatHistory = null;
        return false;
    }

    /// <summary>
    /// Parses collection of <see cref="PromptNode"/> instances and sets output as <see cref="ChatHistory"/>.
    /// </summary>
    /// <param name="nodes">Collection of <see cref="PromptNode"/> to parse.</param>
    /// <param name="chatHistory">Parsing output as <see cref="ChatHistory"/>.</param>
    /// <returns>Returns true if parsing was successful, otherwise false.</returns>
    private static bool TryParse(List<PromptNode> nodes, [NotNullWhen(true)] out ChatHistory? chatHistory)
    {
        chatHistory = null;

        foreach (var node in nodes.Where(IsValidChatMessage))
        {
            (chatHistory ??= []).Add(ParseChatNode(node));
        }

        return chatHistory is not null;
    }

    /// <summary>
    /// Parses a chat node and constructs a <see cref="ChatMessageContent"/> object.
    /// </summary>
    /// <param name="node">The prompt node to parse.</param>
    /// <returns><see cref="ChatMessageContent"/> object.</returns>
    private static ChatMessageContent ParseChatNode(PromptNode node)
    {
        ChatMessageContentItemCollection items = [];
        foreach (var childNode in node.ChildNodes.Where(childNode => childNode.Content is not null))
        {
            if (childNode.TagName.Equals(ImageTagName, StringComparison.OrdinalIgnoreCase))
            {
                if (childNode.Content!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    items.Add(new ImageContent(childNode.Content));
                }
                else
                {
                    items.Add(new ImageContent(new Uri(childNode.Content!)));
                }
            }
            else if (childNode.TagName.Equals(TextTagName, StringComparison.OrdinalIgnoreCase))
            {
                items.Add(new TextContent(childNode.Content));
            }
            else if (childNode.TagName.Equals(AudioTagName, StringComparison.OrdinalIgnoreCase))
            {
                if (childNode.Content!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    items.Add(new AudioContent(childNode.Content));
                }
                else
                {
                    items.Add(new AudioContent(new Uri(childNode.Content!)));
                }
            }
            else if (childNode.TagName.Equals(PdfTagName, StringComparison.OrdinalIgnoreCase))
            {
                if (childNode.Content!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    items.Add(new PdfContent(childNode.Content));
                }
                else
                {
                    items.Add(new PdfContent(new Uri(childNode.Content!)));
                }
            }
            else if (childNode.TagName.Equals(DocxTagName, StringComparison.OrdinalIgnoreCase))
            {
                if (childNode.Content!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    items.Add(new DocxContent(childNode.Content));
                }
                else
                {
                    items.Add(new DocxContent(new Uri(childNode.Content!)));
                }
            }
            else if (childNode.TagName.Equals(DocTagName, StringComparison.OrdinalIgnoreCase))
            {
                if (childNode.Content!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    items.Add(new DocContent(childNode.Content));
                }
                else
                {
                    items.Add(new DocContent(new Uri(childNode.Content!)));
                }
            }
            else if (childNode.TagName.Equals(BinaryTagName, StringComparison.OrdinalIgnoreCase))
            {
                if (childNode.Content!.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                {
                    items.Add(new BinaryContent(childNode.Content));
                }
                else
                {
                    items.Add(new BinaryContent(new Uri(childNode.Content!)));
                }
            }
            else
            {
                throw new NotSupportedException($"Unsupported node type: {childNode.TagName}");
            }
        }

        if (items.Count == 1 && items[0] is TextContent textContent)
        {
            node.Content = textContent.Text;
            items.Clear();
        }

        var authorRole = new AuthorRole(node.Attributes[RoleAttributeName]);

        return items.Count > 0
            ? new ChatMessageContent(authorRole, items)
            : new ChatMessageContent(authorRole, node.Content);
    }

    /// <summary>
    /// Checks if <see cref="PromptNode"/> is valid chat message.
    /// </summary>
    /// <param name="node">Instance of <see cref="PromptNode"/>.</param>
    /// <remarks>
    /// A valid chat message is a node with the following structure:<br/>
    /// TagName = "message"<br/>
    /// Attributes = { "role" : "..." }<br/>
    /// optional one or more child nodes <image>...</image><br/>
    /// optional one or more child nodes <text>...</text>
    /// </remarks>
    private static bool IsValidChatMessage(PromptNode node)
    {
        return
            node.TagName.Equals(MessageTagName, StringComparison.OrdinalIgnoreCase) &&
            node.Attributes.ContainsKey(RoleAttributeName);
    }
}
