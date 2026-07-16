type Identifier = string | number;

export type InstallFormDraft = {
	version: 1;
	policyId: string;
	datasourceLabel: string;
	mappings: Array<{
		sourceId: Identifier;
		dagId: Identifier | null;
		nodeNames: string[];
	}>;
};

type InstallCatalogs = {
	policies: Array<Record<string, unknown>>;
	datasources: Array<Record<string, unknown>>;
	dags: Array<Record<string, unknown>>;
	nodes: Array<Record<string, unknown>>;
};

export type RestoredInstallForm = {
	policyIndex: number | null;
	datasourceIndex: number | null;
	sources: Array<Record<string, unknown>>;
};

const STORAGE_PREFIX = 'dayu:install-form-draft:v1';

function record(value: unknown): Record<string, unknown> | null {
	return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function text(value: unknown): string {
	return typeof value === 'string' ? value.trim() : '';
}

function identifier(value: unknown): Identifier | null {
	if (typeof value === 'string' && value.trim()) return value.trim();
	if (typeof value === 'number' && Number.isSafeInteger(value)) return value;
	return null;
}

function identifierKey(value: unknown): string {
	const normalized = identifier(value);
	return normalized === null ? '' : `${typeof normalized}:${normalized}`;
}

function nodeNames(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return [...new Set(value.map(text).filter(Boolean))];
}

function normalizeDraft(value: unknown): InstallFormDraft | null {
	const source = record(value);
	if (
		!source ||
		source.version !== 1 ||
		typeof source.policyId !== 'string' ||
		typeof source.datasourceLabel !== 'string' ||
		!Array.isArray(source.mappings)
	) {
		return null;
	}
	const mappings = source.mappings;
	return {
		version: 1,
		policyId: text(source.policyId),
		datasourceLabel: text(source.datasourceLabel),
		mappings: mappings.flatMap((value) => {
			const mapping = record(value);
			const sourceId = identifier(mapping?.sourceId);
			if (!mapping || sourceId === null) return [];
			return [
				{
					sourceId,
					dagId: identifier(mapping.dagId),
					nodeNames: nodeNames(mapping.nodeNames),
				},
			];
		}),
	};
}

function browserStorage(): Storage | null {
	try {
		return globalThis.localStorage || null;
	} catch {
		return null;
	}
}

export function installFormDraftKey(namespace: string): string {
	return `${STORAGE_PREFIX}:${encodeURIComponent(text(namespace))}`;
}

export function readInstallFormDraft(
	namespace: string,
	storage: Storage | null = browserStorage()
): InstallFormDraft | null {
	if (!text(namespace) || !storage) return null;
	const key = installFormDraftKey(namespace);
	try {
		const raw = storage.getItem(key);
		if (!raw) return null;
		const draft = normalizeDraft(JSON.parse(raw));
		if (!draft) storage.removeItem(key);
		return draft;
	} catch {
		try {
			storage.removeItem(key);
		} catch {
			// Browser storage is an optional convenience; lifecycle state is server-owned.
		}
		return null;
	}
}

export function writeInstallFormDraft(
	namespace: string,
	draft: InstallFormDraft | null,
	storage: Storage | null = browserStorage()
): boolean {
	if (!text(namespace) || !storage) return false;
	try {
		const key = installFormDraftKey(namespace);
		if (draft) storage.setItem(key, JSON.stringify(draft));
		else storage.removeItem(key);
		return true;
	} catch {
		return false;
	}
}

export function clearInstallFormDraft(namespace: string, storage: Storage | null = browserStorage()): boolean {
	return writeInstallFormDraft(namespace, null, storage);
}

export function createInstallFormDraft(
	policy: Record<string, unknown> | null,
	datasource: Record<string, unknown> | null,
	sources: Array<Record<string, unknown>>
): InstallFormDraft | null {
	const policyId = text(policy?.policy_id);
	const datasourceLabel = text(datasource?.source_label);
	const mappings = (Array.isArray(sources) ? sources : []).flatMap((source) => {
		const sourceId = identifier(source?.id);
		if (sourceId === null) return [];
		return [
			{
				sourceId,
				dagId: identifier(source.dag_selected),
				nodeNames: nodeNames(source.node_selected),
			},
		];
	});
	if (!policyId && !datasourceLabel && mappings.length === 0) return null;
	return { version: 1, policyId, datasourceLabel, mappings };
}

export function restoreInstallFormDraft(
	draft: InstallFormDraft | null,
	catalogs: InstallCatalogs
): RestoredInstallForm {
	const policies = Array.isArray(catalogs.policies) ? catalogs.policies : [];
	const datasources = Array.isArray(catalogs.datasources) ? catalogs.datasources : [];
	const dags = Array.isArray(catalogs.dags) ? catalogs.dags : [];
	const nodes = Array.isArray(catalogs.nodes) ? catalogs.nodes : [];
	const policyIndex = draft?.policyId ? policies.findIndex((item) => text(item.policy_id) === draft.policyId) : -1;
	const datasourceIndex = draft?.datasourceLabel
		? datasources.findIndex((item) => text(item.source_label) === draft.datasourceLabel)
		: -1;
	if (datasourceIndex < 0) {
		return {
			policyIndex: policyIndex >= 0 ? policyIndex : null,
			datasourceIndex: null,
			sources: [],
		};
	}

	const savedMappings = new Map((draft?.mappings || []).map((mapping) => [identifierKey(mapping.sourceId), mapping]));
	const validNodes = new Set(nodes.map((node) => text(node.name)).filter(Boolean));
	const datasourceSources = datasources[datasourceIndex].source_list;
	const sources = (Array.isArray(datasourceSources) ? datasourceSources : []).flatMap((value) => {
		const source = record(value);
		if (!source) return [];
		const mapping = savedMappings.get(identifierKey(source.id));
		const dag =
			mapping?.dagId === null || mapping?.dagId === undefined
				? null
				: dags.find((item) => identifierKey(item.dag_id) === identifierKey(mapping.dagId));
		return [
			{
				...source,
				dag_selected: dag?.dag_id ?? '',
				node_selected: (mapping?.nodeNames || []).filter((nodeName) => validNodes.has(nodeName)),
			},
		];
	});

	return {
		policyIndex: policyIndex >= 0 ? policyIndex : null,
		datasourceIndex,
		sources,
	};
}
