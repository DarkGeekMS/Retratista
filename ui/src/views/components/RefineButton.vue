
<template>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    <base-button type="outline-danger" class="btn-3 generate_button" icon="fa fa-paint-brush" :disabled="!this.firstGen" @click="handleClick" >Refine</base-button>
                </div>
            </div>
        </div>
</template>




<script>
export default {
    name: "refine-button",
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

