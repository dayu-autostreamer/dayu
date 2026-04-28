<template>
	<div class="outline datasource-page">
		<section class="section-card upload-section">
			<div class="section-heading">
				<div>
					<h3>Source Configuration</h3>
					<p class="section-subtitle">Upload datasource manifests, review available inputs, and choose which source the platform should open.</p>
				</div>

				<div class="section-actions">
					<el-tag type="info" size="small" effect="plain">
						<el-icon><Connection /></el-icon>
						{{ sources.length }} configs
					</el-tag>
					<el-tag :type="state === 'open' ? 'success' : 'warning'" size="small" effect="plain">
						<el-icon>
							<VideoPlay v-if="state === 'open'" />
							<VideoPause v-else />
						</el-icon>
						{{ state === 'open' ? 'Datasource Active' : 'Datasource Idle' }}
					</el-tag>
				</div>
			</div>

			<div class="upload-layout">
				<button
					type="button"
					class="upload-dropzone"
					:class="{ 'is-drag-over': isDragOver, 'has-file': pendingFile }"
					@click="triggerFilePicker"
					@dragover.prevent="handleDragOver"
					@dragleave.prevent="handleDragLeave"
					@drop.prevent="handleFileDrop"
				>
					<el-icon class="upload-dropzone__icon">
						<UploadFilled />
					</el-icon>
					<div class="upload-dropzone__title">{{ pendingFile ? 'Ready to Upload' : 'Drop a Source Config Here' }}</div>
					<div class="upload-dropzone__subtitle">
						{{ pendingFile ? pendingFile.name : 'Or click to browse .yaml, .yml, or .json files' }}
					</div>
					<span v-if="pendingFile" class="upload-pill">{{ formatFileSize(pendingFile.size) }}</span>
				</button>

				<div class="upload-panel">
					<div class="upload-panel__header">
						<div class="upload-panel__title">Upload Queue</div>
						<div class="upload-panel__hint">One file at a time for predictable updates</div>
					</div>

					<div class="upload-file-card" :class="{ empty: !pendingFile }">
						<template v-if="pendingFile">
							<div class="upload-file-card__meta">
								<el-icon class="upload-file-card__icon">
									<Document />
								</el-icon>
								<div class="upload-file-card__text">
									<div class="upload-file-card__name">{{ pendingFile.name }}</div>
									<div class="upload-file-card__detail">{{ formatFileSize(pendingFile.size) }} · waiting for upload</div>
								</div>
							</div>
						</template>
						<template v-else>
							<div class="upload-file-card__meta">
								<el-icon class="upload-file-card__icon upload-file-card__icon--muted">
									<FolderOpened />
								</el-icon>
								<div class="upload-file-card__text">
									<div class="upload-file-card__name">No file selected</div>
									<div class="upload-file-card__detail">Choose a datasource manifest to enable upload.</div>
								</div>
							</div>
						</template>
					</div>

					<div class="builder-buttons">
						<el-button round @click="triggerFilePicker">Choose File</el-button>
						<el-button round :disabled="!pendingFile" @click="clearPendingFile">Clear</el-button>
						<el-button type="primary" round :loading="uploadLoading" :disabled="!pendingFile" @click="uploadFile">Upload File</el-button>
					</div>
				</div>
			</div>

			<input ref="fileInput" class="hidden-file-input" type="file" accept=".yaml,.yml,.json" @change="handleFileChange" />
		</section>

		<section class="section-card source-list-section">
			<div class="section-heading">
				<div>
					<h3>Available Source Configurations</h3>
					<p class="section-subtitle">
						{{ selectedSource ? `Selected: ${selectedSource.source_name || selectedSource.source_label}` : 'Pick a configuration card, then open it as the active datasource.' }}
					</p>
				</div>

				<div class="section-actions section-actions--stack">
					<div class="section-actions">
						<el-tag v-if="activeSource" type="success" size="small" effect="plain">
							<el-icon><VideoPlay /></el-icon>
							Running: {{ activeSource.source_name || activeSource.source_label }}
						</el-tag>
						<el-tag v-else type="info" size="small" effect="plain">
							<el-icon><VideoPause /></el-icon>
							No active datasource
						</el-tag>
					</div>

					<div class="builder-buttons">
						<el-button type="primary" round :disabled="state !== 'close' || !selected_label" :loading="loading" @click="submit_query">
							Open Datasource
						</el-button>
						<el-button type="danger" round :disabled="state === 'close'" :loading="kill_loading" @click="stop_query">
							Close Datasource
						</el-button>
					</div>
				</div>
			</div>

			<div v-if="sources.length" class="source-grid">
				<article
					v-for="item in sources"
					:key="item.source_label"
					class="source-card"
					:class="{
						'is-selected': selected_label === item.source_label,
						'is-active': state === 'open' && source_label === item.source_label,
					}"
					role="button"
					tabindex="0"
					@click="selectItem(item)"
					@keydown.enter.prevent="selectItem(item)"
					@keydown.space.prevent="selectItem(item)"
				>
					<div class="source-card__header">
						<div class="source-card__title-group">
							<div class="source-card__title">{{ item.source_name || item.source_label }}</div>
							<div class="source-card__subtitle">{{ item.source_label }}</div>
						</div>

						<div class="source-card__status">
							<el-tag v-if="state === 'open' && source_label === item.source_label" type="success" size="small" effect="plain">Running</el-tag>
							<el-tag v-else-if="selected_label === item.source_label" type="info" size="small" effect="plain">Selected</el-tag>
						</div>
					</div>

					<div class="source-pill-group">
						<span class="source-pill">Type {{ formatValue(item.source_type) }}</span>
						<span class="source-pill">Mode {{ formatValue(item.source_mode) }}</span>
						<span class="source-pill">{{ getSourceList(item).length }} entries</span>
					</div>

					<div class="source-list-block">
						<div class="source-list-block__title">Source List</div>
						<div v-if="getSourceList(item).length" class="source-list-rows">
							<div
								v-for="(source, index) in getSourceList(item)"
								:key="`${item.source_label}-${source.name || 'source'}-${source.url || index}`"
								class="source-row"
							>
								<div class="source-row__main">
									<div class="source-row__name">{{ source.name || `Source ${index + 1}` }}</div>
									<div class="source-row__url">{{ source.url || 'No URL provided' }}</div>
								</div>

								<el-tooltip effect="dark" placement="right" :hide-after="0" popper-class="tooltip-width">
									<template #content>
										<div>Source URL: {{ source.url || 'N/A' }}</div>
										<template v-for="(value, key) in source.metadata || {}" :key="key">
											<div>{{ formatFieldLabel(key) }}: {{ formatValue(value) }}</div>
										</template>
									</template>
									<button type="button" class="details-link" @click.stop>Details</button>
								</el-tooltip>
							</div>
						</div>
						<div v-else class="source-list-empty">No source entries were defined in this configuration.</div>
					</div>

					<div class="source-card__footer">
						<div class="source-card__footer-text">{{ getSourceList(item).length }} source entries ready</div>
						<el-button size="small" type="danger" plain @click.stop="delete_source(item.source_label)">
							<el-icon><Delete /></el-icon>
							Delete
						</el-button>
					</div>
				</article>
			</div>

			<div v-else class="empty-state">
				<el-icon class="empty-state__icon">
					<Document />
				</el-icon>
				<div class="empty-state__title">No source configurations yet</div>
				<div class="empty-state__subtitle">Upload a manifest above to create your first datasource option.</div>
			</div>
		</section>
	</div>
</template>

<script>
import { ElMessage, ElMessageBox } from 'element-plus';
import { Connection, Delete, Document, FolderOpened, UploadFilled, VideoPause, VideoPlay } from '@element-plus/icons-vue';

export default {
	components: {
		Connection,
		Delete,
		Document,
		FolderOpened,
		UploadFilled,
		VideoPause,
		VideoPlay,
	},
	data() {
		return {
			info: [],
			loading: false,
			uploadLoading: false,
			kill_loading: false,
			selected_label: null,
			state: 'close',
			source_label: null,
			pendingFile: null,
			isDragOver: false,
		};
	},
	computed: {
		sources() {
			return Array.isArray(this.info) ? this.info : [];
		},
		selectedSource() {
			return this.sources.find((item) => item.source_label === this.selected_label) || null;
		},
		activeSource() {
			return this.sources.find((item) => item.source_label === this.source_label) || null;
		},
	},
	methods: {
		showMsg(state, msg) {
			ElMessage({
				message: msg,
				showClose: true,
				type: state === 'success' ? 'success' : 'error',
				duration: 3000,
			});
		},
		handleError(error) {
			ElMessage.error('System Error');
			console.error(error);
		},
		formatFieldLabel(key) {
			return String(key || '')
				.replace(/_/g, ' ')
				.replace(/\b\w/g, (letter) => letter.toUpperCase());
		},
		formatValue(value) {
			if (Array.isArray(value)) {
				return value.join(', ');
			}
			if (value && typeof value === 'object') {
				return JSON.stringify(value);
			}
			return value === null || value === undefined || value === '' ? 'N/A' : String(value);
		},
		formatFileSize(size) {
			if (!Number.isFinite(size)) {
				return 'Unknown size';
			}

			if (size < 1024) {
				return `${size} B`;
			}

			if (size < 1024 * 1024) {
				return `${(size / 1024).toFixed(1)} KB`;
			}

			return `${(size / (1024 * 1024)).toFixed(1)} MB`;
		},
		getSourceList(item) {
			return Array.isArray(item?.source_list) ? item.source_list : [];
		},
		triggerFilePicker() {
			const fileInput = this.$refs.fileInput;
			if (!fileInput) {
				return;
			}

			fileInput.value = '';
			fileInput.click();
		},
		setPendingFile(file) {
			if (!file) {
				return;
			}
			this.pendingFile = file;
		},
		handleFileChange(event) {
			const [file] = event.target.files || [];
			this.setPendingFile(file);
		},
		handleDragOver() {
			this.isDragOver = true;
		},
		handleDragLeave() {
			this.isDragOver = false;
		},
		handleFileDrop(event) {
			this.isDragOver = false;
			const [file] = event.dataTransfer?.files || [];
			this.setPendingFile(file);
		},
		clearPendingFile() {
			this.pendingFile = null;
			if (this.$refs.fileInput) {
				this.$refs.fileInput.value = '';
			}
		},
		async uploadFile() {
			if (!this.pendingFile) {
				ElMessage.error('Please choose a source configuration file');
				return;
			}

			this.uploadLoading = true;
			try {
				const formData = new FormData();
				formData.append('file', this.pendingFile);

				const response = await fetch('/api/datasource', {
					method: 'POST',
					body: formData,
				});
				const data = await response.json();
				this.showMsg(data.state, data.msg);

				if (data.state === 'success') {
					this.clearPendingFile();
					await this.getInfo();
					await this.query_state();
				}
			} catch (error) {
				this.handleError(error);
			} finally {
				this.uploadLoading = false;
			}
		},
		async getInfo() {
			try {
				const response = await fetch('/api/datasource');
				const data = await response.json();
				this.info = Array.isArray(data) ? data : [];
			} catch (error) {
				this.handleError(error);
			}
		},
		selectItem(item) {
			this.selected_label = item.source_label;
			localStorage.setItem('source_item', item.source_label);
		},
		async submit_query() {
			if (!this.selected_label) {
				ElMessage.error('Please choose datasource');
				return;
			}

			this.loading = true;
			try {
				const response = await fetch('/api/submit_query', {
					method: 'POST',
					body: JSON.stringify({
						source_label: this.selected_label,
					}),
				});
				const data = await response.json();
				if (data.state === 'success') {
					localStorage.setItem('source_item', this.selected_label);
					await this.query_state();
				}
				this.showMsg(data.state, data.msg);
			} catch (error) {
				this.handleError(error);
			} finally {
				this.loading = false;
			}
		},
		async query_state() {
			try {
				const response = await fetch('/api/query_state');
				const data = await response.json();
				this.state = data.state;
				this.source_label = data.source_label;
			} catch (error) {
				this.handleError(error);
			}
		},
		async stop_query() {
			if (this.state === 'close') {
				return;
			}

			this.kill_loading = true;
			try {
				const response = await fetch('/api/stop_query', {
					method: 'POST',
					body: JSON.stringify({
						source_label: this.source_label,
					}),
				});
				const data = await response.json();
				if (data.state === 'success') {
					this.state = 'close';
					this.selected_label = null;
					this.source_label = null;
					localStorage.removeItem('source_item');
				}
				this.showMsg(data.state, data.msg);
			} catch (error) {
				this.handleError(error);
			} finally {
				this.kill_loading = false;
			}
		},
		async delete_source(sourceLabel) {
			if (this.state === 'open' && this.source_label === sourceLabel) {
				ElMessage.warning('Please close the active datasource before deleting its configuration');
				return;
			}

			try {
				await ElMessageBox.confirm(
					`Delete source configuration "${sourceLabel}"?`,
					'Delete Source Configuration',
					{
						confirmButtonText: 'Delete',
						cancelButtonText: 'Cancel',
						type: 'warning',
					}
				);
			} catch {
				return;
			}

			try {
				const response = await fetch('/api/datasource', {
					method: 'DELETE',
					body: JSON.stringify({
						source_label: sourceLabel,
					}),
				});
				const data = await response.json();

				if (data.state === 'success' && this.selected_label === sourceLabel) {
					this.selected_label = null;
					localStorage.removeItem('source_item');
				}

				this.showMsg(data.state, data.msg);
				if (data.state === 'success') {
					await this.getInfo();
					await this.query_state();
				}
			} catch (error) {
				this.handleError(error);
			}
		},
	},
	async mounted() {
		this.selected_label = localStorage.getItem('source_item');
		await Promise.all([this.query_state(), this.getInfo()]);
	},
};
</script>

<style scoped lang="scss">
.datasource-page {
	padding: 20px;
	display: grid;
	gap: 24px;
	background:
		radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 24%),
		radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.08), transparent 24%);
}

.section-card {
	padding: 24px;
	border-radius: 28px;
	border: 1px solid #e2e8f0;
	background:
		linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.96)),
		#ffffff;
	box-shadow: 0 22px 48px rgba(15, 23, 42, 0.06);
}

.section-heading {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 16px;
	margin-bottom: 20px;
}

.section-heading h3 {
	margin: 0;
	font-size: 24px;
	color: #0f172a;
}

.section-subtitle {
	margin: 8px 0 0;
	font-size: 14px;
	line-height: 1.6;
	color: #64748b;
	max-width: 720px;
}

.section-actions {
	display: flex;
	flex-wrap: wrap;
	align-items: center;
	justify-content: flex-end;
	gap: 8px;
}

.section-actions--stack {
	display: grid;
	justify-items: end;
	gap: 12px;
}

.upload-layout {
	display: grid;
	grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.85fr);
	gap: 18px;
	align-items: stretch;
}

.upload-dropzone {
	border: 1.5px dashed #bfdbfe;
	border-radius: 22px;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.08), transparent 34%),
		linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	min-height: 220px;
	padding: 32px;
	display: grid;
	place-items: center;
	text-align: center;
	gap: 12px;
	cursor: pointer;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.upload-dropzone:hover,
.upload-dropzone.is-drag-over,
.upload-dropzone.has-file {
	border-color: #60a5fa;
	box-shadow: 0 18px 40px rgba(37, 99, 235, 0.12);
	transform: translateY(-1px);
}

.upload-dropzone__icon {
	font-size: 38px;
	color: #2563eb;
}

.upload-dropzone__title {
	font-size: 20px;
	font-weight: 700;
	color: #0f172a;
}

.upload-dropzone__subtitle {
	font-size: 14px;
	line-height: 1.6;
	color: #64748b;
	max-width: 360px;
	overflow-wrap: anywhere;
}

.upload-pill,
.source-pill {
	display: inline-flex;
	align-items: center;
	padding: 6px 10px;
	border-radius: 999px;
	border: 1px solid #dbe4ee;
	background: #ffffff;
	font-size: 12px;
	font-weight: 700;
	color: #334155;
}

.upload-panel {
	display: grid;
	gap: 14px;
	padding: 18px;
	border-radius: 22px;
	border: 1px solid #dbe4ee;
	background: #ffffff;
}

.upload-panel__header {
	display: grid;
	gap: 4px;
}

.upload-panel__title {
	font-size: 16px;
	font-weight: 700;
	color: #0f172a;
}

.upload-panel__hint {
	font-size: 13px;
	color: #64748b;
}

.upload-file-card {
	padding: 16px;
	border-radius: 18px;
	border: 1px solid #dbe4ee;
	background: #f8fafc;
	min-height: 92px;
	display: flex;
	align-items: center;
}

.upload-file-card.empty {
	border-style: dashed;
}

.upload-file-card__meta {
	display: flex;
	align-items: center;
	gap: 12px;
	min-width: 0;
}

.upload-file-card__icon {
	font-size: 24px;
	color: #2563eb;
	flex-shrink: 0;
}

.upload-file-card__icon--muted {
	color: #94a3b8;
}

.upload-file-card__text {
	min-width: 0;
}

.upload-file-card__name {
	font-size: 14px;
	font-weight: 700;
	color: #0f172a;
	overflow-wrap: anywhere;
}

.upload-file-card__detail {
	margin-top: 4px;
	font-size: 13px;
	line-height: 1.5;
	color: #64748b;
}

.builder-buttons {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
}

.hidden-file-input {
	display: none;
}

.source-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
	gap: 16px;
	align-items: start;
}

.source-card {
	display: grid;
	gap: 16px;
	padding: 18px;
	border-radius: 22px;
	border: 1px solid #e2e8f0;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.05), transparent 34%),
		#ffffff;
	box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
	cursor: pointer;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
	min-width: 0;
}

.source-card:hover {
	border-color: #93c5fd;
	box-shadow: 0 18px 40px rgba(37, 99, 235, 0.1);
	transform: translateY(-2px);
}

.source-card.is-selected {
	border-color: #3b82f6;
	box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.16), 0 18px 40px rgba(37, 99, 235, 0.1);
}

.source-card.is-active {
	background:
		linear-gradient(135deg, rgba(16, 185, 129, 0.08), transparent 34%),
		#ffffff;
}

.source-card__header,
.source-card__footer {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 12px;
}

.source-card__title-group {
	display: grid;
	gap: 4px;
	min-width: 0;
}

.source-card__title {
	font-size: 18px;
	font-weight: 700;
	color: #0f172a;
	overflow-wrap: anywhere;
}

.source-card__subtitle,
.source-card__footer-text {
	font-size: 13px;
	line-height: 1.6;
	color: #64748b;
	overflow-wrap: anywhere;
}

.source-card__status {
	flex-shrink: 0;
}

.source-pill-group {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.source-list-block {
	display: grid;
	gap: 12px;
	padding: 14px;
	border-radius: 18px;
	border: 1px solid #e2e8f0;
	background: #f8fafc;
	min-width: 0;
}

.source-list-block__title {
	font-size: 13px;
	font-weight: 700;
	letter-spacing: 0.06em;
	text-transform: uppercase;
	color: #475569;
}

.source-list-rows {
	display: grid;
}

.source-row {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 12px;
	padding: 10px 0;
}

.source-row + .source-row {
	border-top: 1px solid #e2e8f0;
}

.source-row__main {
	min-width: 0;
}

.source-row__name {
	font-size: 14px;
	font-weight: 700;
	color: #0f172a;
	overflow-wrap: anywhere;
}

.source-row__url {
	margin-top: 4px;
	font-size: 12px;
	line-height: 1.6;
	color: #64748b;
	overflow-wrap: anywhere;
}

.details-link {
	border: none;
	background: transparent;
	padding: 0;
	font-size: 12px;
	font-weight: 700;
	color: #2563eb;
	cursor: pointer;
	flex-shrink: 0;
}

.details-link:hover {
	color: #1d4ed8;
}

.source-list-empty {
	font-size: 13px;
	color: #64748b;
	line-height: 1.6;
}

.empty-state {
	min-height: 260px;
	display: grid;
	place-items: center;
	text-align: center;
	border: 1.5px dashed #cbd5e1;
	border-radius: 22px;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.06), transparent 38%),
		linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	padding: 32px;
}

.empty-state__icon {
	font-size: 36px;
	color: #2563eb;
	margin-bottom: 12px;
}

.empty-state__title {
	font-size: 18px;
	font-weight: 700;
	color: #0f172a;
}

.empty-state__subtitle {
	margin-top: 8px;
	font-size: 14px;
	line-height: 1.6;
	color: #64748b;
	max-width: 420px;
}

:deep(.tooltip-width.el-popper) {
	max-width: 320px;
	line-height: 1.6;
}

@media (max-width: 1100px) {
	.upload-layout {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 768px) {
	.datasource-page {
		padding: 14px;
		gap: 16px;
	}

	.section-card {
		padding: 18px;
		border-radius: 20px;
	}

	.section-heading,
	.source-card__header,
	.source-card__footer {
		flex-direction: column;
		align-items: flex-start;
	}

	.section-actions,
	.section-actions--stack {
		justify-items: stretch;
		justify-content: flex-start;
	}

	.source-grid {
		grid-template-columns: 1fr;
	}

	.upload-dropzone,
	.empty-state {
		min-height: 200px;
	}
}
</style>
