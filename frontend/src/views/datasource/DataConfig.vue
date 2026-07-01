<template>
	<div class="outline datasource-page">
		<section class="section-card upload-section">
			<div class="section-heading">
				<div>
					<h3>Upload Source Configuration</h3>
				</div>

				<div class="section-actions">
					<el-tag type="info" size="small" effect="plain">
						<el-icon><Connection /></el-icon>
						{{ sources.length }} configs
					</el-tag>
				</div>
			</div>

			<div
				class="upload-inline-card"
				:class="{ 'is-drag-over': isDragOver, 'has-file': pendingFiles.length }"
				@dragover.prevent="handleDragOver"
				@dragleave.prevent="handleDragLeave"
				@drop.prevent="handleFileDrop"
			>
				<button type="button" class="upload-inline-trigger" @click="triggerFilePicker">
					<el-icon class="upload-inline-trigger__icon">
						<UploadFilled />
					</el-icon>
					<div class="upload-inline-trigger__content">
						<div class="upload-inline-trigger__title">
							{{
								pendingFiles.length === 1
									? pendingFiles[0].name
									: pendingFiles.length
									? `${pendingFiles.length} files selected`
									: 'Select config files'
							}}
						</div>
						<div class="upload-inline-trigger__subtitle">
							{{
								pendingFiles.length
									? `${formatFileSize(pendingFilesTotalSize)} total`
									: 'Drag YAML files here or click to browse'
							}}
						</div>
					</div>
				</button>

				<div class="upload-inline-actions">
					<el-button v-if="pendingFiles.length" text @click="clearPendingFiles">Clear all</el-button>
					<el-button type="primary" round :loading="uploadLoading" :disabled="!pendingFiles.length" @click="uploadFile"
						>Upload</el-button
					>
				</div>
			</div>

			<div v-if="pendingFiles.length" class="upload-queue">
				<div v-for="file in pendingFiles" :key="getPendingFileKey(file)" class="upload-queue__item">
					<div class="upload-queue__text">
						<div class="upload-queue__name">{{ file.name }}</div>
						<div class="upload-queue__meta">{{ formatFileSize(file.size) }}</div>
					</div>
					<button type="button" class="upload-queue__remove" @click="removePendingFile(getPendingFileKey(file))">
						Remove
					</button>
				</div>
			</div>

			<input
				ref="fileInput"
				class="hidden-file-input"
				type="file"
				accept=".yaml,.yml"
				multiple
				@change="handleFileChange"
			/>
		</section>

		<section class="section-card source-list-section">
			<div class="section-heading">
				<div>
					<h3>Available Source Configurations</h3>
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
						<el-button
							type="primary"
							round
							:disabled="state !== 'close' || !selected_label"
							:loading="loading"
							@click="submit_query"
						>
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
							<el-tag
								v-if="state === 'open' && source_label === item.source_label"
								type="success"
								size="small"
								effect="plain"
								>Running</el-tag
							>
							<el-tag v-else-if="selected_label === item.source_label" type="info" size="small" effect="plain"
								>Selected</el-tag
							>
						</div>
					</div>

					<div class="source-pill-group">
						<span class="source-pill">Type: {{ formatValue(item.source_type) }}</span>
						<span class="source-pill">Mode: {{ formatValue(item.source_mode) }}</span>
						<span class="source-pill">{{ getSourceList(item).length }} sources</span>
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
						<div v-else class="source-list-empty">No sources in this configuration.</div>
					</div>

					<div class="source-card__footer">
						<div class="source-card__footer-text">{{ getSourceList(item).length }} sources ready</div>
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
				<div class="empty-state__title">No source configurations</div>
				<div class="empty-state__subtitle">Upload a config file to get started.</div>
			</div>
		</section>
	</div>
</template>

<script>
import { ElMessage } from 'element-plus';
import { Connection, Delete, Document, UploadFilled, VideoPause, VideoPlay } from '@element-plus/icons-vue';

export default {
	components: {
		Connection,
		Delete,
		Document,
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
			pendingFiles: [],
			isDragOver: false,
		};
	},
	computed: {
		sources() {
			return Array.isArray(this.info) ? this.info : [];
		},
		activeSource() {
			return this.sources.find((item) => item.source_label === this.source_label) || null;
		},
		pendingFilesTotalSize() {
			return this.pendingFiles.reduce((total, file) => total + (file.size || 0), 0);
		},
	},
	methods: {
		showMsg(state, msg) {
			ElMessage({
				message: msg,
				showClose: true,
				type: state === 'success' ? 'success' : state === 'partial' ? 'warning' : 'error',
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
		getPendingFileKey(file) {
			return `${file.name}-${file.size}-${file.lastModified}`;
		},
		triggerFilePicker() {
			const fileInput = this.$refs.fileInput;
			if (!fileInput) {
				return;
			}

			fileInput.value = '';
			fileInput.click();
		},
		addPendingFiles(fileList) {
			const files = Array.from(fileList || []).filter(Boolean);
			if (!files.length) {
				return;
			}

			const existingKeys = new Set(this.pendingFiles.map((file) => this.getPendingFileKey(file)));
			const nextPendingFiles = [...this.pendingFiles];

			files.forEach((file) => {
				const key = this.getPendingFileKey(file);
				if (!existingKeys.has(key)) {
					existingKeys.add(key);
					nextPendingFiles.push(file);
				}
			});

			this.pendingFiles = nextPendingFiles;
		},
		handleFileChange(event) {
			this.addPendingFiles(event.target.files);
			event.target.value = '';
		},
		handleDragOver() {
			this.isDragOver = true;
		},
		handleDragLeave() {
			this.isDragOver = false;
		},
		handleFileDrop(event) {
			this.isDragOver = false;
			this.addPendingFiles(event.dataTransfer?.files);
		},
		removePendingFile(fileKey) {
			this.pendingFiles = this.pendingFiles.filter((file) => this.getPendingFileKey(file) !== fileKey);
		},
		clearPendingFiles() {
			this.pendingFiles = [];
			if (this.$refs.fileInput) {
				this.$refs.fileInput.value = '';
			}
		},
		async uploadFile() {
			if (!this.pendingFiles.length) {
				ElMessage.error('Please choose source configuration files');
				return;
			}

			this.uploadLoading = true;
			try {
				const formData = new FormData();
				this.pendingFiles.forEach((file) => {
					formData.append('files', file);
				});

				const response = await fetch('/api/datasource', {
					method: 'POST',
					body: formData,
				});
				const data = await response.json();
				this.showMsg(data.state, data.msg);

				const uploadResults = Array.isArray(data.results) ? data.results : [];
				const succeeded = uploadResults.filter((result) => result.state === 'success');
				const failedNames = new Set(
					uploadResults.filter((result) => result.state !== 'success').map((result) => result.filename)
				);

				if (data.state === 'success') {
					this.clearPendingFiles();
				} else if (data.state === 'partial') {
					this.pendingFiles = this.pendingFiles.filter((file) => failedNames.has(file.name));
				}

				if (succeeded.length > 0 || data.state === 'success') {
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
	background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 24%),
		radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.08), transparent 24%);
}

.section-card {
	padding: 24px;
	border-radius: 28px;
	border: 1px solid #e2e8f0;
	background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.96)), #ffffff;
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

.upload-inline-card {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 16px;
	padding: 16px 18px;
	border: 1.5px dashed #bfdbfe;
	border-radius: 22px;
	background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), transparent 34%),
		linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.upload-inline-card:hover,
.upload-inline-card.is-drag-over,
.upload-inline-card.has-file {
	border-color: #60a5fa;
	box-shadow: 0 14px 30px rgba(37, 99, 235, 0.1);
	transform: translateY(-1px);
}

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

.upload-inline-trigger {
	flex: 1;
	display: flex;
	align-items: center;
	gap: 14px;
	min-width: 0;
	border: none;
	background: transparent;
	padding: 0;
	text-align: left;
	cursor: pointer;
}

.upload-inline-trigger__icon {
	font-size: 28px;
	color: #2563eb;
	flex-shrink: 0;
}

.upload-inline-trigger__content {
	min-width: 0;
}

.upload-inline-trigger__title {
	font-size: 14px;
	font-weight: 700;
	color: #0f172a;
	overflow-wrap: anywhere;
}

.upload-inline-trigger__subtitle {
	margin-top: 4px;
	font-size: 13px;
	line-height: 1.5;
	color: #64748b;
}

.upload-inline-actions {
	display: flex;
	align-items: center;
	flex-wrap: wrap;
	gap: 8px;
	flex-shrink: 0;
}

.upload-queue {
	display: grid;
	gap: 10px;
	margin-top: 14px;
}

.upload-queue__item {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 12px;
	padding: 10px 14px;
	border-radius: 16px;
	border: 1px solid #e2e8f0;
	background: #f8fafc;
}

.upload-queue__text {
	min-width: 0;
}

.upload-queue__name {
	font-size: 13px;
	font-weight: 700;
	color: #0f172a;
	overflow-wrap: anywhere;
}

.upload-queue__meta {
	margin-top: 2px;
	font-size: 12px;
	color: #64748b;
}

.upload-queue__remove {
	border: none;
	background: transparent;
	padding: 0;
	font-size: 12px;
	font-weight: 700;
	color: #2563eb;
	cursor: pointer;
	flex-shrink: 0;
}

.upload-queue__remove:hover {
	color: #1d4ed8;
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
	column-width: 352px;
	column-gap: 16px;
}

.source-card {
	display: inline-grid;
	gap: 16px;
	padding: 18px;
	border-radius: 22px;
	border: 1px solid #e2e8f0;
	background: linear-gradient(135deg, rgba(37, 99, 235, 0.05), transparent 34%), #ffffff;
	box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
	cursor: pointer;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
	width: 100%;
	max-width: 352px;
	min-width: 0;
	margin: 0 0 16px;
	break-inside: avoid;
	-webkit-column-break-inside: avoid;
	page-break-inside: avoid;
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
	background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), transparent 34%), #ffffff;
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
	background: linear-gradient(135deg, rgba(37, 99, 235, 0.06), transparent 38%),
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
	.upload-inline-card {
		flex-direction: column;
		align-items: stretch;
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
		column-count: 1;
		column-width: auto;
	}

	.upload-inline-card,
	.empty-state {
		min-height: 0;
	}
}
</style>
