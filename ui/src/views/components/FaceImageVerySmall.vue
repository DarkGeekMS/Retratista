
<template>
    <section class="section section-components pb-0" id="section-components">
        <div class="container">
            <div class="row justify-content-center">
                
                <div class="col-lg-12">
                    <img v-lazy="'img/sample.png'" v-if="!done"
                                 class="rounded-circle img-fluid shadow shadow-lg--hover"
                                 >
                    <img 
                    v-bind:src="'data:image/jpeg;base64,'+faceimg" v-if="done"
                                 class="rounded-circle img-fluid shadow shadow-lg--hover"
                                 >


                    <!-- loading on face image -->
                    <div v-if="this.done === false">
                        <svg class="loading" width="50%"  viewBox="0 0 100 100">
                        <polyline class="line-cornered" points="0,0 100,0 100,100" stroke-width="10" fill="none"></polyline>
                        <polyline class="line-cornered" points="0,0 0,100 100,100" stroke-width="10" fill="none"></polyline>
                        <polyline class="line-cornered stroke-animation" points="0,0 100,0 100,100" stroke-width="10" fill="none"></polyline>
                        <polyline class="line-cornered stroke-animation" points="0,0 0,100 100,100" stroke-width="10" fill="none"></polyline>
                        </svg>
                    </div>

                </div>
            </div>
        </div>
    </section>

</template>




<script>
import axios from 'axios'


export default {
  name: "face-image-very-small",
  data: function() {
    return{
        faceimg: '',
        done: false
    }
  },
  props: {
    angle: Number,
    lastGeneratedAngle: Number,  
    face: String
    },

  mounted(){
  },
  watch: { 
    lastGeneratedAngle(newVal, oldVal) {
        if (newVal == this.angle){
            console.log(this.angle)

            // var style_ip = 'http://0.0.0.0:5000/rotate'      
            // var rr_ip = 'http://3221ce87d940.ngrok.io/rotate'   
            
            var style_ip = 'http://0.0.0.0:5000/rotate'      
            var rr_ip = 'http://0.0.0.0:5001/rotate'
             
            if (process.env.VUE_APP_MODE == '0'){
                axios.post(style_ip, {angle: this.angle})
                .then(function( response ){
                    this.faceimg = response.data.face;
                    this.done = true;
                    this.$emit('done', this.angle);
                }.bind(this));
            }
            else {
                axios.post(rr_ip, {image: this.face, angle: this.angle})
                .then(function( response ){
                    this.faceimg = response.data.face;
                    this.done = true;
                    this.$emit('done', this.angle);
                }.bind(this));
            }
            
        }
    }
  }

};
</script>
<style>
</style>
