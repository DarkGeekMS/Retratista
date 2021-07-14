
<template>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    <base-button type="outline-info" class="btn-3 generate_button" icon="fa fa-paint-brush" :disabled="!this.firstGen" @click="handleClick" >Head Poses</base-button>
                </div>
            </div>
        </div>
</template>




<script>
export default {
    name: "pose-button",
    props: {
        firstGen: {type: Boolean, default: false},
    },
    methods: {
        handleClick() {
            this.$emit('clicked');
        }
    }
};
</script>

