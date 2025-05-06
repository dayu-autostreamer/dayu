declare module '*.jpg' {
  const src: string;
  export default src;
}

declare module '/images/*' {
  const src: string;
  export default src;
}