<template>
    <div>
        <h3>Data Stream Visualization</h3>
    </div>
    <br>

    <div class="image-container" >
        <!-- 指定宽度 高度 -->
        <div style="text-align: center;">
            <div>
                <!-- 如果imgSrc不存在，不渲染该标签 -->
                <img v-if="imgSrc" :src="imgSrc" style="width: 24vw; height: 32vh;" >
            </div>
     </div>
        <!-- <p>{{ data }}</p> -->
    </div>
    <br>
    <div>
        
        <!-- <p>{{ prompt }}</p> -->
    </div>
</template>

<script>
export default {
    props: {
        data: {
            type: Object,
        },
        prompt: {
            type: String,
        }
    },
    computed: {
        imgSrc() {
            // console.log(this.data)
            if (!this.data) {
                return null
            }
            // 
            else if (!this.data[this.data.length - 1]) {

                return null
            }
            else {
                console.log(this.data[this.data.length - 1])
                return this.data[this.data.length - 1]['visualize']
            }
        }
    }

};
</script>

<style scoped>
.image-container {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>