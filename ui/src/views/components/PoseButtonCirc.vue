
<template>
        <div>
            <div class="row justify-content-center">
                <div class="col-lg-12">

                        <input class='toggle2' id='menu' type='button' style="visibility: hidden;">
                        <label  @click="handleClick" class='style2 background-grad-circbutton' for='menu' style='display: flex; justify-content: center; align-items: center;'>
                            <img src="img/brand/pose.png" style="width: 80%; text-align: center;">
                        </label>
                    
                </div>
            </div>
        </div>

</template>




<script>

export default {
    name: "pose-button-circ",
    methods: {
        handleClick() {
            this.$emit('clicked');
        }
    }
};
</script>

