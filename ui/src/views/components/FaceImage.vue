
<template>
    <section class="section section-components pb-0" id="section-components">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-12">
                    
                    <img v-lazy="'img/sample.png'" v-if="!poses"
                    v-bind:src="'data:image/jpeg;base64,'+face"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 50%;">

                    <img 
                    v-bind:src="'data:image/jpeg;base64,'+face" v-if="poses"
                                 class="rounded-circle img-center img-fluid shadow shadow-lg--hover"
                                 style="width: 50%;">



                    <!-- loading on face image -->
                    <div v-if="this.loading === true">
                        <svg class="loading" width="20%"  viewBox="0 0 100 100">
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
export default {
  name: "face-image",
  props: {
    loading: Boolean,  
    face: String, 
    poses: {
        type: Boolean,
        default: false
    }
    },

};
</script>
<style>
</style>
