<script>
export default {
  props: {
    config: {
      type: Object,
      required: true
    },
    data: {
      type: Array,
      default: () => []
    },
    variableStates: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['update:variable-states'],

  methods: {
    validateData() {
      if (!Array.isArray(this.data)) {
        console.error(`[${this.config.type} Visualization] Invalid data:`, this.data)
        return false
      }
      return true
    },

    updateVariableStates(newStates) {
      this.$emit('update:variable-states', newStates)
    }
  }
}
</script>