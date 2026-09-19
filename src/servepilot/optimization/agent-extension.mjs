/** Only controller-provided tools are registered. No host shell or filesystem tools. */
export default function (pi) {
  const tools = JSON.parse(process.env.OPENBASETEN_AGENT_TOOLS || "[]");
  const base = process.env.OPENBASETEN_AGENT_BRIDGE;
  const token = process.env.OPENBASETEN_AGENT_BRIDGE_TOKEN;
  for (const tool of tools) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      async execute(_id, parameters, signal) {
        const response = await fetch(`${base}/tools/${encodeURIComponent(tool.name)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify(parameters),
          signal,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(JSON.stringify(result));
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: {} };
      },
    });
  }
}
