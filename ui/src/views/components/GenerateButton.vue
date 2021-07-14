
<template>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    <base-button  type="outline-primary" class="btn-3 generate_button" icon="fa fa-paint-brush" :disabled="this.clicked" @click="handleClick" >Generate</base-button>
                </div>
            </div>
        </div>
</template>




<script>
import axios from 'axios'
export default {
    name: "generate-button",
    data: function() {
    return{
        clicked: false
      }
    },
    props: {
        doneGeneration: Boolean 
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

