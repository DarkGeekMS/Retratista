<template>
    <div>
        <hero></hero>
        <generation-module style="margin-bottom:10vw"></generation-module>
    </div>
</template>
<script>
import Hero from "./components/Hero";
import GenerationModule from "./components/GenerationModule";

export default {
  name: "components",
  components: {
    Hero,
    GenerationModule,
    }
};
</script>
