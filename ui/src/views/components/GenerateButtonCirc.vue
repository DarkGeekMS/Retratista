
<template>
        <div>
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    <input class='toggle2' id='menu' @click="handleClick" type='button' style="visibility: hidden;" :disabled="clicked ? true : false">
                    <label  :disabled="this.clicked"  class='style2 background-grad-circbutton' for='menu' style='display: flex; justify-content: center; align-items: center;'>
                        <img src="img/brand/brush3.png" style="width: 80%; text-align: center;">
                    </label>                  
                </div>
            </div>
        </div>
</template>




<script>

export default {
    name: "generate-button-circ",
    data: function() {
    return{
        clicked: false
      }
    },
    props: {
        doneGeneration: Boolean,
    },
    watch: { 
        doneGeneration(newVal, oldVal) {
            console.log(this.clicked)
            this.clicked = !newVal;
        }
    },
    mounted(){
        this.clicked = false;
    },
    methods: {
        handleClick() {
            this.clicked = !this.clicked
            this.$emit('clicked', this.clicked);
        }
    }
};
</script>

